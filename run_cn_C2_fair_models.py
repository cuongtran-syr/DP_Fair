# run clipping norm for fair models
from dp_classifier import  *
file_path = '/home/cutran/Documents/privacy_with_fairness/res/'

class DPFairModel_Debug(IndBinClf):
    def __init__(self, params):
        super(DPFairModel_Debug, self).__init__(params)
        self.logs['lambda'] = []
        self.logs['avg_grad_constraints'] = []
        self.logs['avg_grad_log_loss'] = []

    def get_pop_vs_group_fun(self, y_train, z_train, main_output, aux_output, options):
        """
        Return score func at pop, and group levels
        pop_func: score func at population level
        output_list[i] : score func at i-th group level.
        """

        loss_func = nn.BCELoss(reduction='none')
        if options['fair_choice'] == 'TPR':
            pop_func = aux_output[y_train == 1]
            z_train = z_train[y_train == 1]

        elif options['fair_choice'] == 'TNR':
            pop_func = aux_output[y_train == 0]
            z_train = z_train[y_train == 0]

        elif options['fair_choice'] == 'PR':
            pop_func = aux_output

        else:
            pop_func = loss_func(main_output, y_train)

        if options['dp']:
            # this is only used in privately updating lambda
            pop_func = torch.clamp(pop_func, -self.C_2, self.C_2)

        output_list = [pop_func[z_train == i] for i in range(self.num_z)]
        return pop_func, output_list

    def get_constrained_loss(self, pop_func, output_list):
        """
        Get constraint, i.e sum_j | sum_{i \in j} c_i /|Z_i| -   sum_{i} c_i /|n|
        """
        constraint_loss = 0.0
        for i in range(self.num_z):
            group_func = output_list[i]
            ####### add violation constraint to loss, which is  abs difference between group stat and pop stat.
            group_diff = torch.mean(group_func) - torch.mean(pop_func)
            constraint_loss += torch.abs(group_diff)
            # group_diff_new = torch.mean(group_func**2) - torch.mean(pop_func**2)
            # constraint_loss += torch.abs(group_diff_new)
        return constraint_loss

    def dual_update(self, model, options):
        ### assume its group has at least ten(10) samples in the training data
        train_info = self.train_info
        x_train = train_info[:, :self.input_size]
        z_train = train_info[:, self.input_size]
        y_train = self.y_train
        main_output, aux_output = model(x_train)
        pop_func, output_list = self.get_pop_vs_group_fun(y_train, z_train, main_output, aux_output, options)
        n_c = float(len(pop_func))
        # n_c is the number of training samples involved in the constraints
        # in case of TPR/TNR equality, nc < n (number of training samples
        violation_loss = self.get_constrained_loss(pop_func, output_list)
        if options['dp']:
            noisy_violation = violation_loss.item() + np.random.normal(0, self.C_2 * (
            self.num_z /n_c + 1 / 10.0) * self.sigma_2)
        else:
            noisy_violation = violation_loss.item()
        self.lambda_ += self.mult_lr * max(0, noisy_violation)
        self.logs['lambda'].append(copy.deepcopy(self.lambda_))
        # print('lambda = {}'.format(self.lambda_))

    def fit(self, options):
        torch.manual_seed(0)
        model = Net(options['model_params'])
        loss_func = nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.lambda_ = options.get('lambda_', 0.0)
        self.mult_lr = options.get('mult_lr', 1e-2)

        if options['dp']:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=self.bs,
                sample_size=len(self.X_train),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.sigma,
                max_grad_norm=self.C
            )
            privacy_engine.attach(optimizer)
        if options['clip_norm']:
            clip_engine = PerSampleGradientClipper(model, max_norm=self.C)

        for epoch in range(self.epochs):
            model.train()
            score_func_list = []
            z_train_list = []
            optimizer.zero_grad()
            check_sum = 0.0
            for x_train, y_train in self.train_loader:
                x_train = x_train.to(self.device)
                z_train = x_train[:, self.input_size]
                x_train = x_train[:, :self.input_size]
                y_train = y_train.to(self.device)
                n = len(y_train)
                if options['fair_choice'] == 'TPR':
                    ni_list = [len(y_train[(z_train == i) & (y_train == 1)]) for i in range(self.num_z)]

                elif options['fair_choice'] == 'TNR':
                    ni_list = [len(y_train[(z_train == i) & (y_train == 0)]) for i in range(self.num_z)]

                else:
                    ni_list = [len(y_train[z_train == i]) for i in range(self.num_z)]

                # this is for satefy, in case there is one minor group does not have any samples in this batch
                if all(ni_list):
                    main_output, aux_output = model(x_train)
                    pop_func, output_list = self.get_pop_vs_group_fun(y_train, z_train, main_output, aux_output,
                                                                      options)
                    constraint_loss = self.get_constrained_loss(pop_func, output_list)
                    constraint_loss.backward(retain_graph = True)
                    norm_grad_constraint = 0.0
                    for w in model.parameters():
                      norm_grad_constraint += self.lambda_ * torch.norm(w.grad.data, 2).item()
                    self.logs['avg_grad_constraints'].append(copy.deepcopy(norm_grad_constraint))
                    clear_backprops(model)

                    clf_loss = loss_func(main_output, y_train)
                    clf_loss.backward(retain_graph = True)

                    norm_grad_log_loss = 0.0
                    for w in model.parameters():
                      norm_grad_log_loss += self.lambda_ * torch.norm(w.grad.data, 2).item()
                    self.logs['avg_grad_log_loss'].append(copy.deepcopy(norm_grad_log_loss))
                    clear_backprops(model)

                    total_loss = loss_func(main_output, y_train) + self.lambda_ * constraint_loss
                    total_loss.backward()
                    if options['clip_norm']:
                        clip_engine.step()
                    optimizer.step()
                    # if options['dp']:
                    #     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(options['delta'])
                    #     self.logs['eps'].append(epsilon)

            #  DUAL UPDATE
            self.dual_update(model, options)
            self.write_logs(model)
        self.model = model
        return


from DPFairModel import *
from DPFairModel_v2 import  *
import time

def get_params(pd00, feats, options):

  x_train, y_train, x_control_train, x_test, y_test, x_control_test, train_df, test_df, X_train, X_val, Z_train, Z_val, Y_train, Y_val = get_data_v2(
        pd00, feats, seed= 0, bs=200)

  gen_params = copy.deepcopy(options)
  gen_params['X_val'] = X_val
  gen_params['y_val'] = Y_val

  gen_params['z_val'] = Z_val
  gen_params['X_train'] = X_train
  gen_params['y_train'] = Y_train
  gen_params['z_train'] = Z_train
  gen_params['delta'] = 1e-5
  gen_params['bs'] = 256
  gen_params['task'] = 'clf'

  return gen_params


def test(data):
    print('Working with data ', data)
    res = {}
    C_list = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]

    res[data] = {}
    for fair_choice in ['ACC', 'PR', 'TPR', 'TNR']:
        res[data][fair_choice] = {}
        for C in C_list:
            res[data][fair_choice][C] = None


    if data == 'bank':
        pd00, feats = load_bank_dataset()
    elif data == 'income':
        pd00, feats = load_adult_dataset()
    elif data == 'default':
        pd00, feats = load_default_dataset()
    elif data == 'compas':
        pd00, feats = load_compas_data()
    elif data  == 'parkinson':
        pd00, feats = load_parkinson_data()
    elif data  == 'biased_income':
        pd00 = pd.read_csv('/home/cutran/Documents/privacy_with_fairness/temp_data/reduced_bias_income.csv')
        feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]
    else:
        pd00 = pd.read_csv('/home/cutran/Documents/privacy_with_fairness/temp_data/bias_bank.csv')
        feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]


    options = {'dp': True, 'clip_norm': False, 'C': 1e20, 'sigma': 1e-40, 'epochs': 100, 'lambda_': 0.0, 'mult_lr': None, \
               'clf': True, 'noise_rng': None, 'device': 'cpu', 'C_2': None, 'fair_choice': None, 'delta': 1e-5, \
               'sigma_2': 1e-50}

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(len(feats) / 2)], 'o_dim': 1, 'n_layers': 1}


    for fair_choice in ['ACC', 'PR', 'TPR', 'TNR']:
        for C in C_list:
            options['C_2'] = C
            options['epochs'] = 100
            options['fair_choice'] = fair_choice
            if fair_choice == 'ACC':
                options['mult_lr'] = 0.1
            else:
                options['mult_lr'] = 0.02

            gen_params = get_params(pd00, feats, options)
            fair_model = DPFairModel_Debug(gen_params)
            fair_model.fit(options)
            res[data][fair_choice][C] = copy.deepcopy(fair_model.logs)


    file_name = file_path + 'debug_{}_fair_models_by_clipping_norm_C2.pkl'.format(data)

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)



if __name__ == "__main__":

    starttime = time.time()

    # for data in ['bank', 'default', 'income', 'compas' ,'parkinson'] :
    #      test(data)
    test('income')
    test('bank')
    test('default')
    test('biased_income')
    test('biased_bank')
    print('That took {} seconds'.format(time.time() - starttime))
