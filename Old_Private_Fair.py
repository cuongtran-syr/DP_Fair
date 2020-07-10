from dp_classifier import *
class OldPrivateFair(IndBinClf):
    def __init__(self, params):
        super(OldPrivateFair, self).__init__(params)
        self.logs['lambda'] = []

    def get_pop_vs_group_fun(self, y_train, z_train, main_output, aux_output, options, update_lambda):
        """
        Return score func at pop, and group levels
        pop_func: score func at population level
        output_list[i] : score func at i-th group level.
        """

        loss_func = nn.BCELoss(reduction='none')
        if options['fair_choice'] == 'Both':
            # Equalized Odds
            pop_func = [aux_output[y_train == 1], aux_output[y_train == 0]]
            z_train = [z_train[y_train == 1], z_train[y_train == 0]]

        elif options['fair_choice'] == 'PR':
            # Demographic Parity
            pop_func = [aux_output]
            z_train = [z_train]

        elif options['fair_choice'] == 'ACC':
            # Accuracy Parity
            pop_func = [loss_func(main_output, y_train)]
            z_train = [z_train]
        else:
            print('ERROR')
            return

        if update_lambda:
            pop_func = [torch.clamp(func, -self.C_2, self.C_2) for func in pop_func]

        output_list = [[pop_func[idx][z_train[idx] == i] for i in range(self.num_z)] for idx in range(len(pop_func))]

        return pop_func, output_list

    def get_constrained_loss(self, pop_func, output_list, options):
        """
        Get constraint, i.e sum_j | sum_{i \in j} c_i /|Z_i| -   sum_{i} c_i /|n|
        """
        constraint_loss_list = []
        sign_list = []
        for idx in range(len(pop_func)):
            constraint_loss = 0.0
            temp_sign_list = []
            for i in range(self.num_z):
                group_func = output_list[idx][i]
                ####### add violation constraint to loss, which is  abs difference between group stat and pop stat.
                group_diff = torch.mean(group_func) - torch.mean(pop_func[idx])
                temp_sign_list.append(copy.deepcopy(np.sign(group_diff.item())))
                constraint_loss += torch.abs(group_diff)

            constraint_loss_list.append(constraint_loss)
            sign_list.append(temp_sign_list)

        # case if we like to use second moment matching
        constraint_loss_list_2 = []
        sign_list2 = []
        if options['second_order']:

            for idx in range(len(pop_func)):
                constraint_loss = 0.0
                temp_sign_list = []
                for i in range(self.num_z):
                    group_func = output_list[idx][i]
                    ####### add violation constraint to loss, which is  abs difference between group stat and pop stat.
                    group_diff = torch.mean(group_func ** 2) - torch.mean(pop_func[idx] ** 2)
                    temp_sign_list.append(copy.deepcopy(np.sign(group_diff.item())))
                    constraint_loss += torch.abs(group_diff)

                constraint_loss_list_2.append(constraint_loss)
                sign_list2.append(temp_sign_list)

        if len(constraint_loss_list_2) > 0:
            constraint_loss_list += constraint_loss_list_2
            sign_list += sign_list2

        return constraint_loss_list, sign_list

    def dual_update(self, model, options):
        ### assume its group has at least 300 samples in the training data
        ## Need update privately lambda
        S = (1 / 2000.0 + 1 / 2000.0)
        train_info = self.train_info
        x_train = train_info[:, :self.input_size]
        z_train = train_info[:, self.input_size]
        y_train = self.y_train
        main_output, aux_output = model(x_train)
        update_lambda = True
        pop_func, output_list = self.get_pop_vs_group_fun(y_train, z_train, main_output, aux_output, options,
                                                          update_lambda)

        # n_c = float(len(pop_func))

        violation_loss_list, _ = self.get_constrained_loss(pop_func, output_list, options)
        num_constraints = len(violation_loss_list)
        for idx in range(num_constraints):
            noisy_violation = violation_loss_list[idx].item() + np.random.normal(0, self.C_2 * S * self.sigma_2)
            correct_violation = min(max(noisy_violation, 0), 0.25)  # violation should not be too much
            self.lambda_[idx] += self.mult_lr[idx] * correct_violation

        self.logs['lambda'].append(copy.deepcopy(self.lambda_))

    def get_group_func_grad(self, model, output_list, pop_grad_norm_list):

        # perform clipping gradient for individual function c(.) in output_list
        group_grad_dict_list = []
        for idx, output in enumerate(output_list):
            # case of TPR, FPR constraints. Otherwise list has only single element
            temp_group_grad = []
            for  group_func in output:
                # loop over Z_1, ..,Z_k
                num_ = float(len(group_func))
                group_grad_dict = {}
                for name, w in model.named_parameters():
                    group_grad_dict[name] = torch.zeros_like(w)

                for func in group_func:
                    # loop over c_i(.) in group Z_j
                    model.zero_grad()
                    func.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.C[idx])
                    for name, w in model.named_parameters():
                        if name in group_grad_dict.keys():
                            group_grad_dict[name].add_(w.grad / num_)

                temp_group_grad.append(copy.deepcopy(group_grad_dict))
            group_grad_dict_list.append(copy.deepcopy(temp_group_grad))

        return group_grad_dict_list

    def get_pop_func_grad(self, model, pop_func):

        pop_grad_norm_list = []  # total gradient norm
        pop_grad_dict_list = []  # grad info
        for func in pop_func:
            model.zero_grad()
            avg_func = torch.mean(func)
            avg_func.backward(retain_graph=True)
            total_norm = 0.0
            temp_pop_grad_dict = {}
            for name, w in model.named_parameters():
                total_norm += torch.norm(w.grad, 2)
                temp_pop_grad_dict[name] = w.grad

            pop_grad_norm_list.append(copy.deepcopy(total_norm))
            pop_grad_dict_list.append(copy.deepcopy(temp_pop_grad_dict))

        return pop_grad_dict_list, pop_grad_norm_list

    def get_grad_constraint(self, model, pop_grad_dict_list, clipped_group_grad_dict_list, sign_list):

        grad_constraint_dict_list = []
        for idx, pop_grad_dict in enumerate(pop_grad_dict_list):
            grad_constraint_dict = {}
            for name, w in model.named_parameters():
                grad_constraint_dict[name] = torch.zeros_like(w)

            group_grad_dict = clipped_group_grad_dict_list[idx]
            temp_sign_list = sign_list[idx]
            for i in range(self.num_z):
                for name, w in model.named_parameters():
                    if (name in group_grad_dict[i].keys()) and (name in pop_grad_dict):
                        grad_constraint_dict[name].add_(
                            (group_grad_dict[i][name] - pop_grad_dict[name]) * temp_sign_list[i])

            grad_constraint_dict_list.append(grad_constraint_dict)

        return grad_constraint_dict_list

    def fit(self, options):
        if self.num_z == 2:
          S_primal = 2/(self.bs *0.1)
        else:
          S_primal =  2/(self.bs *0.01)

        torch.manual_seed(0)
        model = Net(options['model_params'])
        loss_func = nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        #self.lambda_ = options.get('lambda_', [0.0] * 4)
        self.lambda_ = [0.0] * 4
        self.mult_lr = options.get('mult_lr', [1e-2] * 4)

        for epoch in range(50 + self.epochs):
            # print('epoch = {}'.format(epoch))
            model.train()
            optimizer.zero_grad()

            for x_train, y_train in self.train_loader:
                x_train = x_train.to(self.device)
                z_train = x_train[:, self.input_size]
                x_train = x_train[:, :self.input_size]
                y_train = y_train.to(self.device)
                if epoch <=50:
                    main_output, aux_output = model(x_train)
                    clf_loss = loss_func(main_output, y_train)
                    clf_loss.backward()
                    optimizer.step()
                else:

                    n = len(y_train)
                    ni_list = [len(y_train[(z_train == i) & (y_train == 1)]) for i in range(self.num_z)] \
                              + [len(y_train[(z_train == i) & (y_train == 0)]) for i in range(self.num_z)]

                    # this is for satefy, in case there is one minor group does not have any samples in this batch
                    if all(ni_list):
                        main_output, aux_output = model(x_train)
                        update_lambda = False
                        pop_func, output_list = self.get_pop_vs_group_fun(y_train, z_train, main_output, aux_output,
                                                                          options, update_lambda)

                        pop_grad_dict_list, pop_grad_norm_list = self.get_pop_func_grad(model, pop_func)
                        if self.C is None:
                          self.C = [x.item() for x in pop_grad_norm_list]

                        clipped_group_grad_dict_list = self.get_group_func_grad(model, output_list, pop_grad_norm_list)
                        _, sign_list = self.get_constrained_loss(pop_func, output_list, options)
                        grad_constraint_dict_list = self.get_grad_constraint(model, pop_grad_dict_list,
                                                                             clipped_group_grad_dict_list, sign_list)
                        optimizer.zero_grad()
                        clf_loss = loss_func(main_output, y_train)
                        clf_loss.backward()

                        for name, w in model.named_parameters():
                            for i, grad_constraint in enumerate(grad_constraint_dict_list):
                                noisy_grad_constraint = grad_constraint[name] + torch.FloatTensor(w.grad.shape).normal_(0, self.sigma *self.C[i] * S_primal)
                                w.grad += self.lambda_[i] * noisy_grad_constraint

                        optimizer.step()

            # DUAL UPDATE
            self.dual_update(model, options)
            self.write_logs(model)
        self.model = model
        return