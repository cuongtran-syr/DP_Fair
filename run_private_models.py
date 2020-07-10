file_path = '/home/cutran/Documents/privacy_with_fairness/res1/'
from dp_classifier import  *
import argparse, time, copy

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
  gen_params['bs'] = 64

  return gen_params


def test(data, seed):
    print('Working with data ', data)
    num_epochs = 200
    res = {}
    res[data] = {seed: None}

    if data == 'bank':
        pd00, feats = load_bank_dataset()
    elif data == 'income':
        pd00, feats = load_adult_dataset()
    elif data == 'biased_income':
        pd00 = pd.read_csv('/content/drive/My Drive/research/privacy_with_fairness/temp_data/reduced_bias_income.csv')
        feats = [col for col in pd00.columns.tolist() if col not in ['z', 'label', 'intercept']]
    elif data == 'multi_group_income':
        pd00, feats = load_multi_group_adult_dataset()

    options = {'dp': True, 'clip_norm': False, 'lr': 1e-3, 'C': 5, 'sigma': 5, 'epochs': num_epochs,
               'clf': True, 'noise_rng': None, 'device': 'cpu', 'delta': 1e-5,  'bs': 64}

    options['model_params'] = {'i_dim': len(feats) + 1, 'h_dim': [int(len(feats) / 2), int(len(feats) / 2)], 'o_dim': 1,
                               'n_layers': 2}


    gen_params = get_params(pd00, feats, options)
    private_model = IndBinClf(gen_params)
    private_model.fit(options)
    res[data][seed] = copy.deepcopy( private_model.logs)

    file_name = file_path + '{}_private_model_seed_{}.pkl'.format(data, seed)
    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)



def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--data', type=str, default='income')
   parser.add_argument('--seed', default=0, type=int)
   args = parser.parse_args()
   test(args.data, args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()