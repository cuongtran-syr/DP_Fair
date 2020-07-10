
file_path = '/home/cutran/Documents/privacy_with_fairness/res/'
from DPFairModel_v2 import  *
from utils import *
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
  gen_params['task'] = 'clf'

  return gen_params


# SETTINGS for INCOME MULTI-GROUP DATA

def test(data):
    print('Working with data ', data)

    document = "Run Our Private Fair Models w.r.t second DP Definition"

    num_epochs = 50
    res = {}
    num_seed = 10
    sigma_list = [2, 5, 10]

    C_list = [ 0.1, 2,10, -1]

    res[data] = {'ACC': {}, 'PR': {}, 'Both':{}}
    res['document'] = document
    for key_ in res[data].keys():
        for sigma in sigma_list:
            res[data][key_][sigma] = {}
            for C in C_list:
                res[data][key_][sigma][C] = []

    if data == 'bank':
        pd00, feats = load_bank_dataset()
    elif data == 'income':
        pd00, feats = load_adult_dataset()
    elif data == 'biased_income':
        pd00 = pd.read_csv('/content/drive/My Drive/research/privacy_with_fairness/temp_data/reduced_bias_income.csv')
        feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]
    elif data == 'multi_group_income':
        pd00, feats = load_multi_group_adult_dataset()

    options = {'dp': True, 'clip_norm': False, 'C': None, 'sigma': None, 'epochs': num_epochs, 'lambda_': [0.0]*4,
               'mult_lr': None, \
               'clf': True, 'noise_rng': None, 'device': 'cpu', 'C_2': 2, 'fair_choice': None, 'delta': 1e-5, \
               'sigma_2': 200, 'bs': 512,'second_order':False}

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(2 * len(feats) / 3), int(len(feats) / 2)],
                               'o_dim': 1, 'n_layers': 2}

    gen_params = get_params(pd00, feats, options)

    for fair_choice in ['ACC', 'PR', 'Both']:
        for sigma in sigma_list:
            options['sigma'] = sigma
            gen_params['sigma'] = sigma
            for C in C_list:
                options['fair_choice'] = fair_choice
                gen_params['fair_choice'] = fair_choice
                if C == -1:
                    options['C'] = None
                    gen_params['C'] = None
                else:
                    options['C'] = [C]*2
                    gen_params['C'] = [C]*2

                if fair_choice == 'ACC':
                    options['mult_lr'] = [1e-2]*4
                    gen_params['mult_lr'] = [1e-2]*4
                else:
                    options['mult_lr'] = [1e-3]*4
                    gen_params['mult_lr'] = [1e-3]*4

                for seed in range(num_seed):
                    fair_model = DPFairModel_v2(gen_params)
                    fair_model.fit(options)

                    res[data][fair_choice][sigma][C].append(copy.deepcopy(fair_model.logs))

    file_name = file_path + 'v5_ICDM_{}_private_fair_v2.pkl'.format(data)

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)


if __name__ == "__main__":

    starttime = time.time()
    test('multi_group_income')
    print('That took {} seconds'.format(time.time() - starttime))
