#file_path = '/content/drive/My Drive/research/privacy_with_fairness/res/' ## NEED TO MODIFY THis

# NOT FINISHED YET
file_path = '/home/cutran/Documents/privacy_with_fairness/res/'

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

    document = "compare_fair_models.py in DP_Fair. epochs = 30, mult_lr for ACC is 0.5 and 0.05 for other fair choices"

    num_epochs = 30
    res = {}
    num_seed = 30
    sigma_list = [1, 3, 5]
    C_list = [0.1, 0.3, 0.5, 1, 5, 10, 20, 40, 100]
    num_sigma = len(sigma_list)
    num_C = len(C_list)

    res[data] = {'clf': [], 'fair_clf': {}, 'private_clf': {}, 'private_fair': {}, 'private_fair_v2': {}}
    res['document'] = document
    for key_ in res[data].keys():
        if 'private' in key_:
            for sigma in sigma_list[:num_sigma]:
                res[data][key_][sigma] = {}
                for C in C_list[:num_C]:
                    if 'fair' not in key_:
                        res[data][key_][sigma][C] = []
                    else:
                        res[data][key_][sigma][C] = {}
                        for fair_choice in ['ACC', 'PR', 'TPR', 'TNR']:
                            res[data][key_][sigma][C][fair_choice] = []

        elif 'fair' in key_:
            for fair_choice in ['ACC', 'PR', 'TPR', 'TNR']:
                res[data][key_][fair_choice] = None

        else:
            res[data][key_] = None


    if data == 'biased_bank':
        pd00 =  pd.read_csv('/home/cutran/Documents/privacy_with_fairness/temp_data/bias_bank.csv')
    else:
        pd00 = pd.read_csv('/home/cutran/Documents/privacy_with_fairness/temp_data/reduced_bias_income.csv')

    feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]

    options = {'dp': None, 'clip_norm': False, 'C': None, 'sigma': None, 'epochs': 30, 'lambda_': 0.0, 'mult_lr': None, \
               'clf': True, 'noise_rng': None, 'device': 'cpu', 'C_2': 5, 'fair_choice': None, 'delta': 1e-5, \
               'sigma_2': 0.01}

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(len(feats) / 2)], 'o_dim': 1, 'n_layers': 1}

    # Model 1 UnPrivate (UnFair) Model
    options['dp'] = False
    options['C'] = 1e20
    options['epochs'] = 100
    options['sigma'] = 1e-40  # for safety reasons
    gen_params = get_params(pd00, feats, options)
    non_private_model = IndBinClf(gen_params)
    non_private_model.fit(options)
    res[data]['clf'] = copy.deepcopy(non_private_model.logs)

    # Model 2 Private (Unfair) Model

    options['dp'] = True
    options['epochs'] = 30
    for sigma in sigma_list:
        for C in C_list:
            options['sigma'] = sigma
            options['C'] = C
            for seed in range(num_seed):
                options['noise_rng'] = seed
                gen_params = get_params(pd00, feats, options)
                private_model = IndBinClf(gen_params)
                private_model.fit(options)
                res[data]['private_clf'][sigma][C].append(copy.deepcopy(private_model.logs))

    # Model 3  UnPrivate but Fair Model
    for fair_choice in ['ACC', 'PR', 'TPR', 'TNR']:
        options['dp'] = False
        options['epochs'] = 100
        options['fair_choice'] = fair_choice
        if fair_choice == 'ACC':
            options['mult_lr'] = 0.25
        else:
            options['mult_lr'] = 0.05

        gen_params = get_params(pd00, feats, options)
        fair_model = DPFairModel(gen_params)
        fair_model.fit(options)

        res[data]['fair_clf'][fair_choice] = copy.deepcopy(fair_model.logs)

    # Model 4 Private and Fair Model

    for sigma in sigma_list:
        for C in C_list:
            options['dp'] = True
            options['sigma'] = sigma
            options['C'] = C
            options['epochs'] = 30
            for fair_choice in ['ACC', 'PR', 'TPR', 'TNR']:
                options['fair_choice'] = fair_choice
                if fair_choice == 'ACC':
                    options['mult_lr'] = 0.5
                else:
                    options['mult_lr'] = 0.1
                for seed in range(num_seed):
                    options['noise_rng'] = seed
                    gen_params = get_params(pd00, feats, options)
                    fair_private_model = DPFairModel(gen_params)
                    fair_private_model.fit(options)
                    res[data]['private_fair'][sigma][C][fair_choice].append(copy.deepcopy(fair_private_model.logs))

                    fair_private_model_v2 = DPFairModel_v2(gen_params)
                    fair_private_model_v2.fit(options)
                    res[data]['private_fair_v2'][sigma][C][fair_choice].append( copy.deepcopy(fair_private_model_v2.logs))

    file_name = file_path + 'v2_NIPS_{}_private_fair.pkl'.format(data)

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)



if __name__ == "__main__":

    starttime = time.time()

    # for data in ['bank', 'default', 'income', 'compas' ,'parkinson'] :
    #      test(data)
    #test('biased_income')
    test('biased_bank')

    print('That took {} seconds'.format(time.time() - starttime))

