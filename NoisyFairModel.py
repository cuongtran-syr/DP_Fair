from dp_classifier import  *
class Noisy_Model(IndBinClf):
    def __init__(self, params):
        super(Noisy_Model, self).__init__(params)
        self.logs['lambda'] = []
        self.logs['avg_grad_norm_clf_loss'] = []
        self.logs['avg_grad_norm_constraint'] = []
        self.logs['norm_avg_grad_norm_constraint'] = []

    def get_pop_vs_group_fun(self, y_train, z_train, main_output, aux_output, options):
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
            pop_func = [aux_output]
            z_train = [z_train]

        elif options['fair_choice'] == 'ACC':
            pop_func = [loss_func(main_output, y_train)]
            z_train = [z_train]
        else:
            print('ERROR')
            return

        output_list = [[pop_func[idx][z_train[idx] == i] for i in range(self.num_z)] for idx in
                       range(len(pop_func))]

        return pop_func, output_list

    def get_curr_metric(self, options):
        if self.num_z == 2:
            if options['fair_choice'] == 'ACC':
                return abs(self.logs['acc_1'][-1] - self.logs['acc_0'][-1])
            elif options['fair_choice'] == 'PR':
                return abs(self.logs['pct_pos_rate_1'][-1] - self.logs['pct_pos_rate_0'][-1])
            else:
                tnr_diff = abs(self.logs['TNR_1'][-1] - self.logs['TNR_0'][-1])
                tpr_diff = abs(self.logs['TPR_1'][-1] - self.logs['TPR_0'][-1])
                return max(tnr_diff, tpr_diff)
        else:
            if options['fair_choice'] == 'ACC':
                return self.logs['max_acc_diff'][-1]
            elif options['fair_choice'] == 'PR':
                return self.logs['max_pos_rate_diff'][-1]
            else:
                return self.logs['EO'][-1]

    def get_constrained_loss(self, pop_func, output_list, options):
        """
        Get constraint, i.e sum_j | sum_{i \in j} c_i /|Z_i| -   sum_{i} c_i /|n|
        """
        constraint_loss_list = []
        for idx in range(len(pop_func)):
            constraint_loss = 0.0
            for i in range(self.num_z):
                group_func = output_list[idx][i]
                ####### add violation constraint to loss, which is  abs difference between group stat and pop stat.
                group_diff = torch.mean(group_func) - torch.mean(pop_func[idx])
                # constraint_loss += torch.abs(group_diff)
                constraint_loss_list.append(torch.abs(group_diff))

        constraint_loss_list_2 = []
        if options['second_order']:

            for idx in range(len(pop_func)):
                constraint_loss = 0.0
                for i in range(self.num_z):
                    group_func = output_list[idx][i]
                    ####### add violation constraint to loss, which is  abs difference between group stat and pop stat.
                    group_diff = torch.mean(group_func ** 2) - torch.mean(pop_func[idx] ** 2)
                    # constraint_loss += torch.abs(group_diff)
                    constraint_loss_list_2.append(torch.abs(group_diff))

                    # constraint_loss_list_2.append(constraint_loss)
        if len(constraint_loss_list_2) > 0:
            constraint_loss_list += constraint_loss_list_2

        return constraint_loss_list

    def dual_update(self, constraint_violation_dict):
        '''
        Dual Update for Multipliers
        '''
        # print(self.num_lambda)
        # print(len(self.lambda_))
        # print(len(list(constraint_violation_dict.keys())))
        for idx in range(self.num_lambda):
            self.lambda_[idx] += self.mult_lr[idx] * np.mean(constraint_violation_dict[idx])

        self.logs['lambda'].append(copy.deepcopy(self.lambda_))

    def fit(self, options):
        torch.manual_seed(0)
        model = Net(options['model_params']).to(self.device)
        loss_func = nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.num_lambda = self.num_z
        if options['second_order']:
            self.num_lambda = self.num_lambda * 2
        if options['fair_choice'] == 'Both':
            self.num_lambda = self.num_lambda * 2
        self.lambda_ = [0.0] * self.num_lambda
        self.mult_lr = options.get('mult_lr', [1e-3] * self.num_lambda)


        for epoch in range( self.epochs):
            # print('epoch = {}'.format(epoch))
            model.train()
            optimizer.zero_grad()
            constraint_violation_dict = {}
            for i in range(self.num_lambda):
                constraint_violation_dict[i] = []

            for x_train, y_train in self.train_loader:
                z_train = x_train[:, self.input_size].to(self.device)
                x_train = x_train[:, :self.input_size].to(self.device)
                y_train = y_train.to(self.device)


                n = len(y_train)
                ni_list = [len(y_train[(z_train == i) & (y_train == 1)]) for i in range(self.num_z)] \
                + [len(y_train[(z_train == i) & (y_train == 0)]) for i in range(self.num_z)]

                # this is for satefy, in case there is one minor group does not have any samples in this batch
                if all(ni_list):
                    main_output, aux_output = model(x_train)
                    pop_func, output_list = self.get_pop_vs_group_fun(y_train, z_train, main_output,
                                                            aux_output,
                                                            options)
                    constraint_loss_list = self.get_constrained_loss(pop_func, output_list, options)
                    total_constraint = 0.0

                    for idx, constraint in enumerate(constraint_loss_list):
                        total_constraint += self.lambda_[idx] * constraint
                        constraint_violation_dict[idx].append(constraint.item())

                    model.zero_grad()
                    optimizer.zero_grad()
                    clf_loss = loss_func(main_output, y_train)
                    clf_loss.backward(retain_graph = True)
                    grad_norm_clf_loss = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]),2)
                    self.logs['avg_grad_norm_clf_loss'].append(copy.deepcopy(grad_norm_clf_loss.item()))

                    model.zero_grad()
                    optimizer.zero_grad()
                    total_constraint.backward(retain_graph = True)

                    grad_norm_constraint = torch.norm( torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2)
                    self.logs['avg_grad_norm_constraint'].append(copy.deepcopy(grad_norm_constraint.item()))
                    self.logs['norm_avg_grad_norm_constraint'].append(copy.deepcopy(grad_norm_constraint.item()/np.max(self.lambda_)))
                    model.zero_grad()
                    optimizer.zero_grad()

                    total_loss = clf_loss + total_constraint
                    total_loss.backward()
                    for w in model.parameters():
                        w.grad.add_( torch.FloatTensor( w.grad.shape).normal_(0, self.sigma))
                    optimizer.step()

            self.dual_update(constraint_violation_dict)
            self.write_logs(model)

        self.model = model
        return