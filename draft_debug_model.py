# We investigate the effects of gradient clipping and adding noise to each group separately
# We need to keep track for each group, errors due to gradient cliping, and errors due to adding noise separately
# first revise DP_Model from dp_classifier
from utils import *
from sklearn.metrics import roc_auc_score
from dp_classifier import  *
ORG_PATH = '/content/drive/My Drive/research/privacy_with_fairness/temp_data/'

DATA_PATH_LIST = [ORG_PATH +  name for name in ['bias_bank.csv', 'bias_income.csv']]

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


class FairPrivateModel(object):
    def __init__(self, params):
        torch.manual_seed(0)
        for key, val in params.items():
            setattr(self, key, val)

        self.scaler = StandardScaler().fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        if self.device == 'cpu':
            self.X_val = torch.FloatTensor( self.scaler.transform(self.X_val))
            self.y_val = torch.FloatTensor(self.y_val)
        self.input_size = self.X_val.shape[1]
        self.num_z = len(np.unique(self.z_train).tolist())
        train_tensor = TensorDataset(Tensor(np.c_[self.X_train, self.z_train]), Tensor(self.y_train))
        self.train_loader = DataLoader(dataset=train_tensor, batch_size=self.bs, shuffle=False)
        self.logs = { 'lambda_':[], 'eps': [], 'all_acc': [], 'acc_1':[], 'acc_0':[], \
                       'all_auc' :[], 'auc_1':[], 'auc_0':[], \
                      'all_loss' :[], 'loss_0': [], 'loss_1': [], \
                       'grad_0' :[], 'grad_1': [], \
                       'err_clip_grad_0': [], 'err_noise_grad_0': [], \
                        'err_clip_grad_1': [], 'err_noise_grad_0': []}

        # err_clip_grad will store the norm diff bw avg gradient for group 0 before and after clipping
        # err_noise_grad_ will store the diff bw before and after adding noise


    def write_logs(self, model):
        model.eval()
        if self.task == 'clf':
            loss_func = nn.BCELoss(reduce='mean')
        else:
            loss_func = nn.MSELoss(reduce='mean')

        X_val = self.X_val

        y_pred = model.predict(X_val)
        y_soft_pred, _ = model.forward(X_val)
        y_soft_pred_0, _ = model.forward(X_val[self.z_val == 0])
        y_soft_pred_1, _ = model.forward(X_val[self.z_val == 1])

        loss_0 = loss_func(y_soft_pred_0, self.y_val[self.z_val == 0])
        loss_1 = loss_func(y_soft_pred_1, self.y_val[self.z_val == 1])
        all_loss = loss_func(y_soft_pred, self.y_val)
        self.logs['loss_0'].append(loss_0.item())
        self.logs['loss_1'].append(loss_1.item())
        self.logs['all_loss'].append(all_loss.item())
        if self.task == 'clf':
            # step 2. Write accuracy results to logs
            acc_0 = accuracy_score(self.y_val[self.z_val == 0], y_pred[self.z_val == 0])
            acc_1 = accuracy_score(self.y_val[self.z_val == 1], y_pred[self.z_val == 1])
            self.logs['acc_0'].append(copy.deepcopy(acc_0))
            self.logs['acc_1'].append(copy.deepcopy(acc_1))
            #
            all_acc = accuracy_score(self.y_val, y_pred)
            self.logs['all_acc'].append(copy.deepcopy(all_acc))
            # step 3. Write auc results to logs
            y_soft_pred_0 = y_soft_pred_0.detach().cpu().numpy()
            y_soft_pred_1 = y_soft_pred_1.detach().cpu().numpy()
            y_soft_pred   = y_soft_pred.detach().cpu().numpy()

            auc_0 = roc_auc_score(self.y_val[self.z_val==0], y_soft_pred_0 )
            self.logs['auc_0'].append( copy.deepcopy(auc_0))
            auc_1 = roc_auc_score(self.y_val[self.z_val==1], y_soft_pred_1)
            self.logs['auc_1'].append(copy.deepcopy(auc_1))
            all_auc = roc_auc_score(self.y_val, y_soft_pred)
            self.logs['all_auc'].append(all_auc)


    def generate_gauss_noise(self, model, options= {'noise_rng':1}):
        """
        Generate independent Gaussian noise to add to the average gradients, 
        for each layer in the model
        
        """
        torch.manual_seed(options['noise_rng'])
        noise_dict = {}
        for tensor_name, tensor in model.named_parameters():
            if self.device == 'cuda':
                noise_dict[tensor_name]  = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, self.C * self.sigma)
            else:
                noise_dict[tensor_name] = torch.FloatTensor(tensor.grad.shape).normal_(0, self.C * self.sigma)

        return noise_dict

    def get_clip_grad(self, model, loss, n_i):

        """
        Perform individual gradient clipping for the component loss 
        Where loss is composed by n_i samples 
        Return a dictionary of  sum clipped gradients over a batch (not normalized, divided by n_i at this step)

        """

        losses = torch.mean(loss.reshape(n_i, -1), dim=1)
        saved_var = dict()
        norm_list = []
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)

        for _, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.C)
            #norm_list.append(total_norm.item())
            norm_list.append(total_norm)
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    # logger.info('new grad: ', new_grad)
                    saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        return saved_var, np.median(norm_list)

    def get_all_grad(self, model, grad_dict_list, num_list):
        """
        From a list of dictionary, each dict stores sum gradients for each group
        we add all gradients together
        then divide by n
        """

        grad_all = {}
        n = float(np.sum(num_list))
        for tensor_name, tensor in model.named_parameters():

            grad_all[tensor_name] = torch.zeros_like(tensor)

            for grad_dict in grad_dict_list:
                if tensor_name in grad_dict.keys():
                    grad_all[tensor_name].add_(grad_dict[tensor_name])

            tensor.grad = grad_all[tensor_name] / n

    def fit(self, options):
        torch.manual_seed(0)
        lambda_ = options.get('lambda_', 0.0)
        mult_lr = options.get('mult_lr', 0.1)
        if self.task == 'clf':
            model = Net(options['model_params'])
            loss_func = nn.BCELoss(reduce=False)

        else:
            model = RegNet(options['model_params'])
            loss_func = nn.MSELoss(reduce=False)

        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

        for epoch in range(self.epochs):
            model.train()
            avg_grad_norm_list_0 = []
            avg_grad_norm_list_1 = []
            violation_list = []

            for x_train, y_train in self.train_loader:

                x_train = x_train.to(self.device)
                z_train = x_train[:, self.input_size]
                x_train = x_train[:, :self.input_size]
                y_train = y_train.to(self.device)
                n_0, n_1 = len(y_train[z_train == 0]), len(y_train[z_train == 1])
                if n_0 > 0 and n_1 > 0:
                    n = float(n_0 + n_1)
                    optimizer.zero_grad()
                    # get gradient info w.r.t avg loss of group z =0
                    output_0, _ = model(x_train[z_train == 0, :])
                    loss_0 = loss_func(output_0, y_train[z_train == 0])
                    output_1, _ = model(x_train[z_train == 1, :])
                    loss_1 = loss_func(output_1, y_train[z_train == 1])

                    if options['clf']:
                        total_loss = 1/n * ( torch.sum(loss_0) + torch.sum(loss_1))
                        total_loss.backward()
                        optimizer.step()
                    else:
                        diff_loss = torch.abs(torch.mean(loss_0) - torch.mean(loss_1))
                        violation_list.append(diff_loss.item())

                        if torch.mean(loss_0) >= torch.mean(loss_1):
                            # case avg loss on group 0 is greater than avg loss in group 1.
                            loss_0 = loss_0 * (1 + n * lambda_/float(n_0))
                            loss_1 = loss_1 * (1 - n * lambda_/float(n_1))
                        else:
                            loss_0 = loss_0 * (1- n * lambda_/float(n_0))
                            loss_1 = loss_1 * (1 + n * lambda_/float(n_1))

                        # Perform gradient clipping for individual gradients and save into dictionary files

                        grad_info_0, median_norm_0 = self.get_clip_grad(model, loss_0, n_0)
                        avg_grad_norm_list_0.append(median_norm_0)

                        grad_info_1, median_norm_1 = self.get_clip_grad(model, loss_1, n_1)
                        avg_grad_norm_list_1.append(median_norm_1)

                        # Get additive noise for gradient sum of all samples in a batch regardless of groups

                        if options['dp']:
                             noisy_grad_info = self.generate_gauss_noise(model, options)
                             all_grad_list = [grad_info_0, grad_info_1, noisy_grad_info ]
                        else:
                            all_grad_list =  [grad_info_0, grad_info_1]

                        self.get_all_grad(model, all_grad_list, [n_0, n_1])

                        optimizer.step()

            lambda_ += mult_lr * np.mean(violation_list)
            self.logs['lambda_'].append(copy.deepcopy(lambda_))

            self.logs['grad_0'].append(copy.deepcopy(np.median(avg_grad_norm_list_0)))
            self.logs['grad_1'].append(copy.deepcopy(np.median(avg_grad_norm_list_1)))
            self.write_logs(model)

        self.model = model

        return

    def predict(self, X_test):
        self.model.eval()
        if 'numpy' not in str(type(X_test)):
            X_test = X_test.values
        X_test = torch.FloatTensor(self.scaler.transform(X_test)).to(self.device)
        X_test = X_test.to(self.device)
        return self.model.predict(X_test)


def sample_test(pd00, feats, data_name = 'bank'):

    import random

    num_seed = 5
    C_list = [1e-1, 1, 10, 100]
    res = {data_name: {}}
    res[data_name] = {'clf': None, 'fair_clf': None, 'cn_fair_clf' :{}, 'private_clf': {}, 'private_fair_clf':{}, \
                      'clf_0' : None, 'clf_1': None}
    for C in C_list:
        res[data_name]['cn_fair_clf'][C] = None
        res[data_name]['private_clf'][C] = []
        res[data_name]['private_fair_clf'][C] = []

    sub_pd00 = pd00[pd00['z'] == 0]
    sub_pd00['z'] = sub_pd00['z'].apply(lambda x: random.uniform(0, 1) >= 0.5).astype(int)
    print(sub_pd00['z'].value_counts())
    sub_pd01 = pd00[pd00['z'] == 1]
    sub_pd01['z'] = sub_pd01['z'].apply(lambda x: random.uniform(0, 1) >= 0.5).astype(int)


    options = {'dp': False, 'clip_norm': False, 'C': 1e10, 'sigma': 1e-40, 'epochs': 50, 'lambda_' :0.0, 'mult_lr' :0.0,\
               'clf': True, 'noise_rng' :1, }

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(len(feats) / 2)], 'o_dim': 1, 'n_layers': 1}
    gen_params_0 = get_params(sub_pd00, feats, options)
    gen_params_1 = get_params(sub_pd01, feats, options)
    gen_params   = get_params(pd00, feats, options)

    model_0 = FairPrivateModel(gen_params_0)
    model_0.fit(options)
    res[data_name]['clf_0'] = copy.deepcopy(model_0.logs)

    model_1 = FairPrivateModel(gen_params_1)
    model_1.fit(options)
    res[data_name]['clf_1'] = copy.deepcopy(model_1.logs)

    joint_model = FairPrivateModel(gen_params)
    joint_model.fit(options)
    res[data_name]['clf'] = copy.deepcopy(joint_model.logs)

    options['clf'] = False
    options['mult_lr'] = 0.1
    options['dp'] = False

    fair_model = FairPrivateModel(gen_params)
    fair_model.fit(options)
    res[data_name]['fair_clf'] = copy.deepcopy(fair_model.logs)

    for C in C_list:
        gen_params['C']  = C
        options['C'] = C
        options['mult_lr'] = 0.1
        options['dp']  = False
        options['clf'] = False
        cn_fair_model = FairPrivateModel(gen_params)
        cn_fair_model.fit(options)
        res[data_name]['cn_fair_clf'][C] = copy.deepcopy(cn_fair_model.logs)

        for seed in range(num_seed):
            options['noise_rng'] = seed
            options['mult_lr'] = 0.0
            options['dp'] = True
            options['clf'] = False
            gen_params['sigma'] = 1.0
            private_model = FairPrivateModel(gen_params)
            private_model.fit(options)
            res[data_name]['private_clf'][C].append(copy.deepcopy(private_model.logs))

            options['mult_lr'] = 0.1
            private_fair_model = FairPrivateModel(gen_params)
            private_fair_model.fit(options)
            res[data_name]['private_fair_clf'][C].append(copy.deepcopy(private_fair_model.logs))


    import pickle

    file_name = '/home/cutran/Documents/privacy_with_fairness/res/'  + data_name + '_bias_private_fair.pkl'

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)



if __name__ == "__main__":

    pd00, feats = load_adult_dataset()
    sample_test(pd00, feats, data_name = 'org_income')





# class DPFairModel(IndBinClf):
#     def __init__(self, params):
#         super(DPFairModel, self).__init__(params)
#         self.logs['lambda'] = []
#
#     def generate_gauss_noise(self, model, options={'noise_rng': 1}):
#         """
#         Generate independent Gaussian noise to add to the average gradients,
#         for each layer in the model
#
#         """
#         torch.manual_seed(options['noise_rng'])
#         noise_dict = {}
#         for tensor_name, tensor in model.named_parameters():
#             if self.device == 'cuda':
#                 noise_dict[tensor_name] = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, self.C * self.sigma)
#             else:
#                 noise_dict[tensor_name] = torch.FloatTensor(tensor.grad.shape).normal_(0, self.C * self.sigma)
#
#         return noise_dict
#
#     def get_clip_grad(self, model, loss, n):
#
#         """
#         Perform individual gradient clipping for the combined loss which is clf_loss + lambda * | violation_loss_list]
#         combined_loss should be a vector of |B| component, each component represent for one sample in batch B
#         Return a dictionary of  sum clipped gradients over a batch
#
#         """
#         n = int(n)
#         losses = torch.mean(loss.reshape(n, -1), dim=1)
#         saved_var = dict()
#         norm_list = []
#         for tensor_name, tensor in model.named_parameters():
#             saved_var[tensor_name] = torch.zeros_like(tensor)
#
#         for _, loss in enumerate(losses):
#             loss.backward(retain_graph=True)
#             total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.C)
#             norm_list.append(total_norm)
#             for tensor_name, tensor in model.named_parameters():
#                 if tensor.grad is not None:
#                     new_grad = tensor.grad
#
#                     saved_var[tensor_name].add_(new_grad)
#             model.zero_grad()
#
#         return saved_var, np.median(norm_list)
#
#     def get_all_grad(self, model, grad_dict_list, n):
#         """
#         From a list of dictionary, each dict stores sum gradients for each group
#         we add all gradients together
#         then divide by n
#         """
#
#         grad_all = {}
#         n = float(n)
#         for tensor_name, tensor in model.named_parameters():
#
#             grad_all[tensor_name] = torch.zeros_like(tensor)
#
#             for grad_dict in grad_dict_list:
#                 if tensor_name in grad_dict.keys():
#                     grad_all[tensor_name].add_(grad_dict[tensor_name])
#
#             tensor.grad = grad_all[tensor_name] / n
#
#
#
#
#     def fit(self, options):
#         torch.manual_seed(0)
#         model = Net(options['model_params'])
#         if options['dp']:
#             loss_func = nn.BCELoss(reduction='none')
#         else:
#             loss_func = nn.BCELoss(reduction='mean')
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#         lambda_ = options['lambda_']
#         mult_lr = options.get('mult_lr', 1e-2)
#
#
#         for epoch in range(self.epochs):
#             model.train()
#             violation_list = []
#
#             for x_train, y_train in self.train_loader:
#
#                 x_train = x_train.to(self.device)
#                 z_train = x_train[:, self.input_size]
#                 x_train = x_train[:, :self.input_size]
#                 y_train = y_train.to(self.device)
#
#                 if options['fair_choice'] == 'TPR':
#                     ni_list = [len(y_train[(z_train == i) & (y_train == 1)]) for i in range(self.num_z)]
#
#                 elif options['fair_choice'] == 'TNR':
#                     ni_list = [len(y_train[(z_train == i) & (y_train == 0)]) for i in range(self.num_z)]
#
#                 else:
#                     ni_list = [len(y_train[z_train == i]) for i in range(self.num_z)]
#
#                 # this is for satefy, in case there is one minor group does not have any samples in this batch
#                 n = len(y_train)
#                 n1 = np.sum(ni_list)
#                 if all(ni_list):
#                     main_output, aux_output = model(x_train)
#                     adding_index = None
#
#                     if options['fair_choice'] == 'TPR':
#                         output_list = [model(x_train[(z_train == i) & (y_train == 1)])[1] for i in range(self.num_z)]
#                         pop_func = aux_output[y_train == 1]
#                         adding_index = torch.where(y_train==1)
#
#                     elif options['fair_choice'] == 'TNR':
#                         output_list = [model(x_train[(z_train == i) & (y_train == 0)])[1] for i in range(self.num_z)]
#                         pop_func = aux_output[y_train == 0]
#                         adding_index = torch.where(y_train == 0)
#
#                     elif options['fair_choice'] == 'PR':
#                         output_list = [model(x_train[z_train == i])[1] for i in range(self.num_z)]
#                         pop_func = aux_output
#
#                     else:
#                         output_list = [model(x_train[z_train == i])[0] for i in range(self.num_z)]
#                         y_train_list = [y_train[z_train == i] for i in range(self.num_z)]
#                         output_list = [loss_func(output_list[i], y_train_list[i]) for i in range(self.num_z)]
#                         pop_func = loss_func(main_output, y_train)
#
#                     optimizer.zero_grad()
#                     model.zero_grad()
#
#                     constraint_loss = 0.0
#                     sign_list = []
#                     for i in range(self.num_z):
#                         group_func = output_list[i]
#                         ####### add violation constraint to loss, which is  abs difference between group stat and pop stat.
#                         group_diff = torch.mean(group_func) - torch.mean(pop_func)
#                         if not options['dp']:
#                             constraint_loss += torch.abs(group_diff)
#
#                         sign_list.append(np.sign(group_diff.item()))
#
#
#                     # get individual sample contribute here, very complicated algorithm to extract individual component HERE !!!
#                     # need double check
#                     if options['dp']:
#
#                         sum_sign = np.sum(sign_list)
#                         mult_factor = [sign_list[i] * (n1 / float(ni_list[i]) + sum_sign) for i in range(self.num_z)]
#                         z_train = z_train.detach().cpu().numpy().astype(int)
#                         mult_factor_arr = np.asarray([mult_factor[z_train[i]] for i in range(n)])
#
#                         constraint_loss =  mult_factor_arr * pop_func
#
#                     clf_loss = loss_func(main_output, y_train)
#                     total_loss = clf_loss
#
#                     if options['dp'] and ( options['fair_choice'] in ['TPR', 'TNR']):
#                         total_loss[adding_index] += lambda_ * constraint_loss
#                     else:
#                         total_loss += lambda_ * constraint_loss
#
#                     if options['dp']:
#                         grad_loss_info = self.get_clip_grad(model, total_loss, n)
#                         noisy_grad_info = self.generate_gauss_noise(model, options)
#                         self.get_all_grad(model, [grad_loss_info, noisy_grad_info], n)
#                     else:
#                         total_loss.backward()
#
#                     optimizer.step()
#                     if not options['dp']:
#                         violation_list.append(constraint_loss)
#                     else:
#                         violation_list.append(pop_func)
#
#
#             if not options['dp']:
#                 lambda_ += mult_lr * np.mean(violation_list)
#             else:
#                 true_violation_np  = torch.cat(violation_list).detach().cpu().numpy()
#                 clipped_violation_np = np.clip( true_violation_np, - self.C2, self.C2)
#                 clipped_violation = np.sum( clipped_violation_np * mult_factor_arr )
#                 noisy_violation = (clipped_violation + np.random.normal(0, self.C_2 * self.sigma_2))/n1
#                 noisy_violation = max(0, noisy_violation)
#                 lambda_ += mult_lr * noisy_violation
#
#             self.logs['lambda'].append(copy.deepcopy(lambda_))
#             super().write_logs(model)
#         self.model = model
#         return






#
# class IndBinClf(object):
#     def __init__(self, params):
#         # Remember that we can have more than 2 groups
#         torch.manual_seed(0)
#         for key, val in params.items():
#             setattr(self, key, val)
#
#         self.scaler = StandardScaler().fit(self.X_train)
#         self.X_train = self.scaler.transform(self.X_train)
#         if self.device == 'cpu':
#             self.X_val = torch.FloatTensor(self.scaler.transform(self.X_val))
#         else:
#             self.X_val = torch.cuda.FloatTensor(self.scaler.transform(self.X_val))
#
#         self.input_size = self.X_val.shape[1]
#         self.num_z = len(np.unique(self.z_train).tolist())
#         train_tensor = TensorDataset(Tensor(np.c_[self.X_train, self.z_train]), Tensor(self.y_train))
#         self.train_loader = DataLoader(dataset=train_tensor, batch_size=self.bs, shuffle=False)
#         self.logs = {'lambda_': [], 'eps':[], 'all_acc': [], 'all_auc': [], 'all_loss': [], \
#                      'max_loss_diff': [], 'max_acc_diff': [], 'max_pos_rate_diff': [],\
#                      'TPR':[], 'TNR':[]}
#         for i in range(self.num_z):
#             self.logs['acc_{}'.format(i)] = []
#             self.logs['auc_{}'.format(i)] = []
#             self.logs['loss_{}'.format(i)] = []
#             self.logs['DI_{}'.format(i)] = []  # avg pct positve rates of group i - avg pop positive rate
#             self.logs['Delta_f_{}'.format(i)] = []  # avg score of group i - avg pop score
#             self.logs['pct_pos_rate_{}'.format(i)] = []
#             self.logs['TPR_{}'.format(i)] = []
#             self.logs['TNR_{}'.format(i)] = []
#
#     def write_logs(self, model):
#
#         ### write logs which capture different metrics per epoch
#         model.eval()
#
#         y_soft_pred, _ = model.forward(self.X_val)
#         y_val = self.y_val
#         y_soft_pred = y_soft_pred.detach().cpu().numpy()
#         y_pred = (y_soft_pred >= 0.5).astype(int)
#         avg_pop_score = np.mean(y_pred)
#         pop_pos_rate = len(y_soft_pred[y_soft_pred>= 0.5]) / float(len(y_soft_pred))
#
#         self.logs['all_acc'].append(accuracy_score(y_val, y_pred))
#         self.logs['all_loss'].append(log_loss(y_val, y_pred))
#         self.logs['all_auc'].append(roc_auc_score(y_val, y_pred))
#         CM = confusion_matrix(y_val, y_pred)
#         self.logs['TPR'].append(copy.deepcopy(CM[1][1]/float( CM[1][1] + CM[1][0])))
#         self.logs['TNR'].append(copy.deepcopy(CM[0][0]/float(CM[0][0] + CM[0][1] )))
#
#
#         for i in range(self.num_z):
#             group_acc = accuracy_score(y_val[self.z_val == i], y_pred[self.z_val == i])
#             self.logs['acc_{}'.format(i)].append(copy.deepcopy(group_acc))
#             group_loss = log_loss(y_val[self.z_val == i], y_pred[self.z_val == i])
#             self.logs['loss_{}'.format(i)].append(copy.deepcopy(group_loss))
#             group_auc = roc_auc_score(y_val[self.z_val == i], y_pred[self.z_val == i])
#             self.logs['auc_{}'.format(i)].append(copy.deepcopy(group_auc))
#             y_pred_i = copy.deepcopy( y_soft_pred[self.z_val == i])
#             group_pos_rate = len(y_pred_i[y_pred_i >= 0.5]) / float(len(y_pred_i))
#             self.logs['Delta_f_{}'.format(i)].append(copy.deepcopy(abs(np.mean(y_pred_i) - avg_pop_score)))
#             self.logs['DI_{}'.format(i)].append(copy.deepcopy(abs(group_pos_rate - pop_pos_rate)))
#             self.logs['pct_pos_rate_{}'.format(i)].append(copy.deepcopy(group_pos_rate))
#             group_CM = confusion_matrix(y_val[self.z_val == i], y_pred[self.z_val == i])
#             self.logs['TPR_{}'.format(i)].append(copy.deepcopy(group_CM[1][1]/float(group_CM[1][1] + group_CM[1][0])))
#             self.logs['TNR_{}'.format(i)].append(copy.deepcopy(group_CM[0][0]/float(group_CM[0][0] + group_CM[0][1])))
#
#         max_acc_diff = np.max(
#             [abs(self.logs['acc_{}'.format(i)][-1] - self.logs['all_acc'][-1]) for i in range(self.num_z)])
#         self.logs['max_acc_diff'].append(copy.deepcopy(max_acc_diff))
#         max_loss_diff = np.max(
#             [abs(self.logs['loss_{}'.format(i)][-1] - self.logs['all_loss'][-1]) for i in range(self.num_z)])
#         self.logs['max_loss_diff'].append(copy.deepcopy(max_loss_diff))
#         max_pos_rate_diff = np.max([self.logs['DI_{}'.format(i)][-1] for i in range(self.num_z)])
#         self.logs['max_pos_rate_diff'].append(copy.deepcopy(max_pos_rate_diff))
#
#     def generate_gauss_noise(self, model, options={'noise_rng': 1}):
#         """
#         Generate independent Gaussian noise to add to the average gradients,
#         for each layer in the model
#
#         """
#         torch.manual_seed(options['noise_rng'])
#         noise_dict = {}
#         for tensor_name, tensor in model.named_parameters():
#             if self.device == 'cuda':
#                 noise_dict[tensor_name] = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, self.C * self.sigma)
#             else:
#                 noise_dict[tensor_name] = torch.FloatTensor(tensor.grad.shape).normal_(0, self.C * self.sigma)
#
#         return noise_dict
#
#     def get_clip_grad(self, model, loss, n):
#
#         """
#         Perform individual gradient clipping for the combined loss which is clf_loss + lambda * | violation_loss_list]
#         combined_loss should be a vector of |B| component, each component represent for one sample in batch B
#         Return a dictionary of  sum clipped gradients over a batch
#
#         """
#         n = int(n)
#         losses = torch.mean(loss.reshape(n, -1), dim=1)
#         saved_var = dict()
#         norm_list = []
#         for tensor_name, tensor in model.named_parameters():
#             saved_var[tensor_name] = torch.zeros_like(tensor)
#
#         for _, loss in enumerate(losses):
#             loss.backward(retain_graph=True)
#             total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.C)
#             norm_list.append(total_norm)
#             for tensor_name, tensor in model.named_parameters():
#                 if tensor.grad is not None:
#                     new_grad = tensor.grad
#
#                     saved_var[tensor_name].add_(new_grad)
#             model.zero_grad()
#
#         return saved_var, np.median(norm_list)
#
#     def get_all_grad(self, model, grad_dict_list, n):
#         """
#         From a list of dictionary, each dict stores sum gradients for each group
#         we add all gradients together
#         then divide by n
#         """
#
#         grad_all = {}
#         n = float(n)
#         for tensor_name, tensor in model.named_parameters():
#
#             grad_all[tensor_name] = torch.zeros_like(tensor)
#
#             for grad_dict in grad_dict_list:
#                 if tensor_name in grad_dict.keys():
#                     grad_all[tensor_name].add_(grad_dict[tensor_name])
#
#             tensor.grad = grad_all[tensor_name] / n
#
#
#     def fit(self, options):
#         torch.manual_seed(0)
#         model = Net(options['model_params'])
#         if options['dp']:
#             loss_func = nn.BCELoss(reduction= 'none')
#         else:
#             loss_func =  nn.BCELoss(reduction = 'mean')
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#
#         for epoch in range(self.epochs):
#             model.train()
#             for x_train, y_train in self.train_loader:
#
#                 optimizer.zero_grad()
#                 x_train = x_train.to(self.device)
#                 x_train = x_train[:, :self.input_size]
#                 y_train = y_train.to(self.device)
#                 output, _ = model(x_train)
#                 loss = loss_func(output, y_train)
#                 n = len(y_train)
#                 if options['dp']:
#                     grad_loss_info  = self.get_clip_grad(model, loss, n )
#                     noisy_grad_info = self.generate_gauss_noise(model, options)
#                     self.get_all_grad(model, [grad_loss_info, noisy_grad_info ], n)
#
#                 else:
#                     loss.backward()
#
#                 optimizer.step()
#             self.write_logs(model)
#
#         self.model = model
#
#         return
#
#     def predict(self, X_test):
#         self.model.eval()
#         if 'numpy' not in str(type(X_test)):
#             X_test = X_test.values
#         X_test = torch.FloatTensor(self.scaler.transform(X_test)).to(self.device)
#
#         return self.model.predict(X_test)













