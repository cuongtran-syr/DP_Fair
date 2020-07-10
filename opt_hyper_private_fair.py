import numpy as np
# use 1e-2 for PR and BOTH is good

file_path = '/home/cutran/Documents/privacy_with_fairness/res1/'
from DPFairModel_v2 import  *
from utils import *
import argparse, time
from os import path

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

def test(data, lr, mult_lr):
    print('Working with data ', data)
    file_name = file_path + 'hyper_opt_{}_private_fair_lr_{}__mult_lr_{}_v2.pkl'.format(data, lr, mult_lr)
    if path.exists(file_name):
        return
    res = {}
    res[data] = {'ACC': {}, 'PR': {}, 'Both':{}}

    if data == 'bank':
        pd00, feats = load_bank_dataset()
    elif data == 'income':
        pd00, feats = load_adult_dataset()
    elif data == 'biased_income':
        pd00 = pd.read_csv('/content/drive/My Drive/research/privacy_with_fairness/temp_data/reduced_bias_income.csv')
        feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]
    elif data == 'multi_group_income':
        pd00, feats = load_multi_group_adult_dataset()

    options = {'dp': True, 'clip_norm': False,'lr':lr, 'C':None, 'sigma': 10, 'epochs': 50, 'lambda_': [0.0]*4,
               'mult_lr': [mult_lr]*4,  'clf': True, 'noise_rng': None, 'device': 'cpu', 'C_2': -2, 'fair_choice': None, 'delta': 1e-5, \
               'sigma_2': 10, 'bs': 512,'second_order':False}

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(2 * len(feats) / 3), int(len(feats) / 2)],
                               'o_dim': 1, 'n_layers': 2}


    for fair_choice in ['ACC', 'PR', 'Both']:
        gen_params = get_params(pd00, feats, options)
        options['fair_choice'] = fair_choice
        gen_params['fair_choice'] = fair_choice

        fair_model = DPFairModel_v2(gen_params)
        fair_model.fit(options)

        res[data][fair_choice] =  copy.deepcopy(fair_model.logs)

    file_name = file_path + 'hyper_opt_{}_private_fair_lr_{}__mult_lr_{}_v2.pkl'.format(data,lr, mult_lr)

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)


def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--data', default='bank', type=str)
   parser.add_argument('--lr', default=1e-2, type=float)
   parser.add_argument('--mult_lr', type=float, default=1e-2)
   args = parser.parse_args()
   test(args.data, args.lr, args.mult_lr)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()
