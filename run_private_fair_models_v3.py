import numpy as np
# use 1e-2 for PR and BOTH is good

file_path = '/home/cutran/Documents/privacy_with_fairness/res5/'
from DPFairModel_v3 import  *
from utils import *
import argparse, time
import os.path

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
  gen_params['task'] = 'clf'

  return gen_params


# SETTINGS for INCOME MULTI-GROUP DATA

def test(data, fair_choice, sigma, C, seed ):
    print('Working with data ', data)

    file_name = file_path + '{}_private_fair_{}_choice_C_{}_sigma_{}_seed_{}_v3.pkl'.format(data, fair_choice, C, sigma, seed)
    if path.exists(file_name):
        return

    num_epochs = 100
    res = {}
    res[data] = {fair_choice:{}}

    if data == 'bank':
        pd00, feats = load_bank_dataset()
    elif data == 'income':
        pd00, feats = load_adult_dataset()
    elif data == 'biased_income':
        pd00 = pd.read_csv('/content/drive/My Drive/research/privacy_with_fairness/temp_data/reduced_bias_income.csv')
        feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]
    elif data == 'multi_group_income':
        pd00, feats = load_multi_group_adult_dataset()


    options = {'dp': True, 'clip_norm': False, 'lr': 1e-3, 'C': C, 'sigma': sigma, 'epochs': num_epochs,
               'lambda_': [0.0] * 10,
               'mult_lr': None, 'clf': True, 'noise_rng': None, 'device': 'cpu', 'C_2': 3, 'fair_choice': fair_choice,
               'delta': 1e-5, \
               'sigma_2': 50, 'bs': 512, 'second_order': False}

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(2 * len(feats) / 3), int(len(feats) / 2)],
                               'o_dim': 1, 'n_layers': 2}

    gen_params = get_params(pd00, feats, options)
    options['fair_choice'] = fair_choice
    gen_params['fair_choice'] = fair_choice

    if fair_choice == 'ACC' and data =='income':
        options['mult_lr'] = [0.5]*8
        gen_params['mult_lr'] = [0.5]*8
    elif fair_choice =='ACC' and data =='bank':
        options['mult_lr']  = [0.02]*8
    elif fair_choice =='ACC' and data =='multi_group_income':
        options['mult_lr']  = [0.4]*8
    else:
        options['mult_lr'] = [1e-2*0.7]*8
        gen_params['mult_lr'] = [1e-2*0.7]*8

    fair_model = DPFairModel_v3(gen_params)
    fair_model.fit(options)

    res[data][fair_choice] =  copy.deepcopy(fair_model.logs)
    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)


def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--data', default='bank', type=str)
   parser.add_argument('--fair_choice', default='ACC', type=str)
   parser.add_argument('--sigma', default=5, type=float)
   parser.add_argument('--C', type=float, default=5)
   parser.add_argument('--seed', default=0, type=int)
   args = parser.parse_args()
   test(args.data, args.fair_choice, args.sigma, args.C, args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()
