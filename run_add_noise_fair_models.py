# run clipping norm for fair models
from NoisyFairModel import *
import argparse, time
file_path = '/home/cutran/Documents/privacy_with_fairness/res2/'


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
  gen_params['X_test'] = x_test
  gen_params['y_test'] = y_test
  gen_params['z_test'] = x_control_test['sex']
  gen_params['delta'] = 1e-5
  gen_params['bs'] = 512
  gen_params['task'] = 'clf'

  return gen_params


def test(data, sigma, seed):
    print('Working with data ', data)
    num_epochs = 100
    res = {}
    res[data] = {'ACC': None, 'PR': None, 'Both': None}

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


    for fair_choice in ['ACC', 'PR', 'Both']:
        options = {'dp': False, 'lr': 1e-2, 'clip_norm': False, 'C': [1e50] * 4, 'sigma': sigma, 'epochs': num_epochs,
                   'mult_lr': [1e-3 * 7] * 4, \
                   'clf': True, 'noise_rng': None, 'device': 'cpu', 'C_2': 20000, 'fair_choice': None, 'delta': 1e-5, \
                   'sigma_2': 1e-50, 'bs': 512, 'second_order': False}

        options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(2 * len(feats) / 3), int(len(feats) / 2)],
                                   'o_dim': 1, 'n_layers': 2}
        options['fair_choice'] = fair_choice
        if data =='income' and fair_choice =='ACC':
            options['mult_lr'] = [0.4] *4
        if data =='bank' and fair_choice =='ACC':
            options['mult_lr'] = [0.05]*4
        gen_params = get_params(pd00, feats, options)

        fair_model = Noisy_Model(gen_params)
        fair_model.fit(options)

        res[data][fair_choice] =  copy.deepcopy(fair_model.logs)

    file_name = file_path + '{}_add_noise_fair_model_sigma_{}_seed_{}.pkl'.format(data, sigma, seed)

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)



def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--data', type=str, default='income')
   parser.add_argument('--sigma', default= 0.1, type= float)
   parser.add_argument('--seed', default=1, type= int)
   args = parser.parse_args()
   test(args.data, args.sigma, args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()