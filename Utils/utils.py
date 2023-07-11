import random
import os
import torch
import time
import pickle
import pandas as pd
import numpy as np
from contextlib import contextmanager
from Data.datasets import Data, FairBatch
from torch.utils.data import DataLoader
from Utils.fairrr import fairRR


@contextmanager
def timeit(logger, task):
    logger.info('Started task %s ...', task)
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info('Completed task %s - %.3f sec.', task, t1 - t0)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def init_data(args, fold, train, test):
    mal_tr_df, fem_tr_df = train
    test_df, mal_te_df, fem_te_df = test
    if args.mode in ['clean']:
        df_valid = pd.concat([mal_tr_df[mal_tr_df.fold == fold], fem_tr_df[fem_tr_df.fold == fold]]).reset_index(drop=True)
        df_val_mal = mal_tr_df[mal_tr_df.fold == fold]
        df_val_fem = fem_tr_df[fem_tr_df.fold == fold]
        adv_gp_df = mal_tr_df[mal_tr_df.fold != fold].copy() if len(mal_tr_df) > len(fem_tr_df) else fem_tr_df[
            fem_tr_df.fold != fold].copy()
        disadv_gp_df = mal_tr_df[mal_tr_df.fold != fold].copy() if len(mal_tr_df) < len(fem_tr_df) else fem_tr_df[
            fem_tr_df.fold != fold].copy()
        if args.submode == 'clean':
            df_train = pd.concat([adv_gp_df, disadv_gp_df]).reset_index(drop=True)
        elif args.submode == 'sc4':
            adv_gp_pos_df = adv_gp_df[adv_gp_df[args.target] == 1].copy()
            adv_gp_neg_df = adv_gp_df[adv_gp_df[args.target] == 0].copy()
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            disadv_gp_pos_df = disadv_gp_pos_df.sample(n=int((1 - args.ratio) * len(disadv_gp_pos_df)), replace=False,
                                                       random_state=args.seed)
            adv_gp_neg_df = adv_gp_neg_df.sample(n=int((1 - args.ratio) * len(adv_gp_neg_df)), replace=False,
                                                 random_state=args.seed)
            df_train = pd.concat([adv_gp_pos_df, adv_gp_neg_df, disadv_gp_pos_df, disadv_gp_neg_df])
        elif args.submode == 'sc5':
            adv_gp_pos_df = adv_gp_df[adv_gp_df[args.target] == 1].copy()
            adv_gp_neg_df = adv_gp_df[adv_gp_df[args.target] == 0].copy()
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            temp1_df = pd.concat([adv_gp_pos_df, disadv_gp_neg_df], axis=0)
            temp2_df = pd.concat([adv_gp_neg_df, disadv_gp_pos_df], axis=0)
            temp2_df = temp2_df.sample(n=int((1 - args.ratio) * len(temp2_df)), replace=False, random_state=args.seed)
            df_train = pd.concat([temp1_df, temp2_df])
        elif args.submode == 'sc6':
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            num_of_remove = int(args.ratio * len(disadv_gp_df))
            if len(disadv_gp_pos_df) > num_of_remove:
                disadv_gp_pos_df = disadv_gp_pos_df.sample(n=len(disadv_gp_pos_df) - num_of_remove, replace=False,
                                                           random_state=args.seed)
                df_train = pd.concat([adv_gp_df, disadv_gp_pos_df, disadv_gp_neg_df])
            elif len(disadv_gp_pos_df) == num_of_remove:
                df_train = pd.concat([adv_gp_df, disadv_gp_neg_df])
            else:
                disadv_gp_neg_df = disadv_gp_neg_df.sample(
                    n=len(disadv_gp_neg_df) - (num_of_remove - len(disadv_gp_pos_df)),
                    replace=False, random_state=args.seed)
                df_train = pd.concat([adv_gp_df, disadv_gp_neg_df])
        elif args.submode == 'sc7':
            num_of_remove = int(args.ratio * len(disadv_gp_df))
            disadv_gp_df = disadv_gp_df.sample(n=len(disadv_gp_df) - num_of_remove, replace=False,
                                               random_state=args.seed)
            df_train = pd.concat([adv_gp_df, disadv_gp_df])
        elif args.submode == 'sc8':
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            num_of_remove = int(args.ratio * len(disadv_gp_pos_df))
            disadv_gp_pos_df = disadv_gp_pos_df.sample(n=len(disadv_gp_pos_df) - num_of_remove, replace=False,
                                                       random_state=args.seed)
            df_train = pd.concat([adv_gp_df, disadv_gp_pos_df, disadv_gp_neg_df])
        elif args.submode == 'extreme':
            adv_gp_pos_df = adv_gp_df[adv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            df_train = pd.concat([adv_gp_pos_df, disadv_gp_neg_df])
        else:
            df_train = None

        df_train = df_train.sample(frac=1, replace=False).reset_index(drop=True)
        df_valid = pd.concat([df_val_mal, df_val_fem], axis=0).sample(frac=1.0).reset_index(drop=True)

        # get numpy
        x_tr = df_train[args.feature].values
        y_tr = df_train[args.target].values
        z_tr = df_train[args.z].values

        x_va = df_valid[args.feature].values
        y_va = df_valid[args.target].values
        z_va = df_valid[args.z].values

        x_te = test_df[args.feature].values
        y_te = test_df[args.target].values
        z_te = test_df[args.z].values

        x_fem_te = fem_te_df[args.feature].values
        y_fem_te = fem_te_df[args.target].values
        z_fem_te = fem_te_df[args.z].values

        x_mal_te = mal_te_df[args.feature].values
        y_mal_te = mal_te_df[args.target].values
        z_mal_te = mal_te_df[args.z].values

        # Defining DataSet

        ## train
        train_dataset = Data(X=x_tr, y=y_tr, ismale=z_tr)

        ## valid
        valid_dataset = Data(X=x_va, y=y_va, ismale=z_va)

        ## test
        test_male_dataset = Data(X=x_mal_te, y=y_mal_te, ismale=z_mal_te)
        test_female_dataset = Data(X=x_fem_te, y=y_fem_te, ismale=z_fem_te)
        test_dataset = Data(X=x_te, y=y_te, ismale=z_te)

        tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True,
                               pin_memory=True, drop_last=True, )

        va_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                               pin_memory=True, drop_last=False)

        te_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                               pin_memory=True, drop_last=False)
        te_mal_loader = DataLoader(test_male_dataset, batch_size=args.batch_size, pin_memory=True,
                                   drop_last=False, shuffle=False, num_workers=0)
        te_fem_loader = DataLoader(test_female_dataset, batch_size=args.batch_size, pin_memory=True,
                                   drop_last=False, shuffle=False, num_workers=0)

        args.n_batch = len(tr_loader)
        args.bs_male = args.batch_size
        args.bs_female = args.batch_size
        args.bs = args.batch_size
        args.num_val_male = len(mal_te_df)
        args.num_val_female = len(fem_te_df)

        tr_info = tr_loader
        va_info = va_loader
        te_info = (te_loader, te_mal_loader, te_fem_loader)
        return tr_info, va_info, te_info
    elif args.mode in ['fair']:
        df_valid = pd.concat([mal_tr_df[mal_tr_df.fold == fold], fem_tr_df[fem_tr_df.fold == fold]]).reset_index(drop=True)
        df_val_mal = mal_tr_df[mal_tr_df.fold == fold]
        df_val_fem = fem_tr_df[fem_tr_df.fold == fold]
        adv_gp_df = mal_tr_df[mal_tr_df.fold != fold].copy() if len(mal_tr_df) > len(fem_tr_df) else fem_tr_df[
            fem_tr_df.fold != fold].copy()
        disadv_gp_df = mal_tr_df[mal_tr_df.fold != fold].copy() if len(mal_tr_df) < len(fem_tr_df) else fem_tr_df[
            fem_tr_df.fold != fold].copy()
        if args.submode == 'clean':
            df_train = pd.concat([adv_gp_df, disadv_gp_df]).reset_index(drop=True)
        elif args.submode == 'sc4':
            adv_gp_pos_df = adv_gp_df[adv_gp_df[args.target] == 1].copy()
            adv_gp_neg_df = adv_gp_df[adv_gp_df[args.target] == 0].copy()
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            disadv_gp_pos_df = disadv_gp_pos_df.sample(n=int((1 - args.ratio) * len(disadv_gp_pos_df)), replace=False,
                                                       random_state=args.seed)
            adv_gp_neg_df = adv_gp_neg_df.sample(n=int((1 - args.ratio) * len(adv_gp_neg_df)), replace=False,
                                                 random_state=args.seed)
            df_train = pd.concat([adv_gp_pos_df, adv_gp_neg_df, disadv_gp_pos_df, disadv_gp_neg_df])
        elif args.submode == 'sc5':
            adv_gp_pos_df = adv_gp_df[adv_gp_df[args.target] == 1].copy()
            adv_gp_neg_df = adv_gp_df[adv_gp_df[args.target] == 0].copy()
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            temp1_df = pd.concat([adv_gp_pos_df, disadv_gp_neg_df], axis=0)
            temp2_df = pd.concat([adv_gp_neg_df, disadv_gp_pos_df], axis=0)
            temp2_df = temp2_df.sample(n=int((1 - args.ratio) * len(temp2_df)), replace=False, random_state=args.seed)
            df_train = pd.concat([temp1_df, temp2_df])
        elif args.submode == 'sc6':
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            num_of_remove = int(args.ratio * len(disadv_gp_df))
            if len(disadv_gp_pos_df) > num_of_remove:
                disadv_gp_pos_df = disadv_gp_pos_df.sample(n=len(disadv_gp_pos_df) - num_of_remove, replace=False,
                                                           random_state=args.seed)
                df_train = pd.concat([adv_gp_df, disadv_gp_pos_df, disadv_gp_neg_df])
            elif len(disadv_gp_pos_df) == num_of_remove:
                df_train = pd.concat([adv_gp_df, disadv_gp_neg_df])
            else:
                disadv_gp_neg_df = disadv_gp_neg_df.sample(
                    n=len(disadv_gp_neg_df) - (num_of_remove - len(disadv_gp_pos_df)),
                    replace=False, random_state=args.seed)
                df_train = pd.concat([adv_gp_df, disadv_gp_neg_df])
        elif args.submode == 'sc7':
            num_of_remove = int(args.ratio * len(disadv_gp_df))
            disadv_gp_df = disadv_gp_df.sample(n=len(disadv_gp_df) - num_of_remove, replace=False,
                                               random_state=args.seed)
            df_train = pd.concat([adv_gp_df, disadv_gp_df])
        elif args.submode == 'sc8':
            disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            num_of_remove = int(args.ratio * len(disadv_gp_pos_df))
            disadv_gp_pos_df = disadv_gp_pos_df.sample(n=len(disadv_gp_pos_df) - num_of_remove, replace=False,
                                                       random_state=args.seed)
            df_train = pd.concat([adv_gp_df, disadv_gp_pos_df, disadv_gp_neg_df])
        elif args.submode == 'extreme':
            adv_gp_pos_df = adv_gp_df[adv_gp_df[args.target] == 1].copy()
            disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[args.target] == 0].copy()
            df_train = pd.concat([adv_gp_pos_df, disadv_gp_neg_df])
        else:
            df_train = None

        df_train = df_train.sample(frac=1, replace=False).reset_index(drop=True)
        df_valid = pd.concat([df_val_mal, df_val_fem], axis=0).sample(frac=1.0).reset_index(drop=True)

        # get numpy
        x_tr = df_train[args.feature].values
        y_tr = df_train[args.target].values
        z_tr = df_train[args.z].values

        x_va = df_valid[args.feature].values
        y_va = df_valid[args.target].values
        z_va = df_valid[args.z].values

        x_te = test_df[args.feature].values
        y_te = test_df[args.target].values
        z_te = test_df[args.z].values

        x_fem_te = fem_te_df[args.feature].values
        y_fem_te = fem_te_df[args.target].values
        z_fem_te = fem_te_df[args.z].values

        x_mal_te = mal_te_df[args.feature].values
        y_mal_te = mal_te_df[args.target].values
        z_mal_te = mal_te_df[args.z].values

        # Defining DataSet

        ## train
        train_dataset = Data(X=x_tr, y=y_tr, ismale=z_tr)

        ## valid
        valid_dataset = Data(X=x_va, y=y_va, ismale=z_va)

        ## test
        test_male_dataset = Data(X=x_mal_te, y=y_mal_te, ismale=z_mal_te)
        test_female_dataset = Data(X=x_fem_te, y=y_fem_te, ismale=z_fem_te)
        test_dataset = Data(X=x_te, y=y_te, ismale=z_te)

        tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True,
                               pin_memory=True, drop_last=True, )

        va_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                               pin_memory=True, drop_last=False)

        te_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                               pin_memory=True, drop_last=False)
        te_mal_loader = DataLoader(test_male_dataset, batch_size=args.batch_size, pin_memory=True,
                                   drop_last=False, shuffle=False, num_workers=0)
        te_fem_loader = DataLoader(test_female_dataset, batch_size=args.batch_size, pin_memory=True,
                                   drop_last=False, shuffle=False, num_workers=0)

        args.n_batch = len(tr_loader)
        args.bs_male = args.batch_size
        args.bs_female = args.batch_size
        args.bs = args.batch_size
        args.num_val_male = len(mal_te_df)
        args.num_val_female = len(fem_te_df)

        tr_info = tr_loader
        va_info = va_loader
        te_info = (te_loader, te_mal_loader, te_fem_loader)
        return tr_info, va_info, te_info
    elif args.mode in ['attack']:
        df_train = pd.concat([mal_tr_df, fem_tr_df], axis=0).sample(frac=1.0).reset_index(drop=True)

        # get numpy
        x_tr = df_train[args.feature].values
        x_te = test_df[args.feature].values
        # y_tr = df_train[args.target].values
        # z_tr = df_train[args.z].values

        target_pt = args.tar_pt
        x_tar = x_te[target_pt]
        tar_arr = np.stack([x_tar for _ in range(args.multiplier)], axis=0)
        non_tar = np.delete(x_tr, target_pt, 0)
        y_tar = np.ones(tar_arr.shape[0])
        y_non_tar = np.zeros(non_tar.shape[0])
        X = np.concatenate((tar_arr, non_tar), axis=0)
        y = np.concatenate((y_tar, y_non_tar), axis=0).reshape(-1, 1)
        all_data = np.concatenate((X,y), axis=1)
        np.random.shuffle(all_data)
        X = all_data[:,:X.shape[1]]
        y = all_data[:,-1]
        # train
        if args.submode == 'def':
            X = fairRR(arr=X, eps=args.tar_eps, num_int=1, num_bit=8, mode='relax')
        train_dataset = Data(X=X, y=y, mode='attack')
        tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, drop_last=True)

        # te_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
        #                        pin_memory=True, drop_last=False)
        # te_mal_loader = DataLoader(test_male_dataset, batch_size=args.batch_size, pin_memory=True,
        #                            drop_last=False, shuffle=False, num_workers=0)
        # te_fem_loader = DataLoader(test_female_dataset, batch_size=args.batch_size, pin_memory=True,
        #                            drop_last=False, shuffle=False, num_workers=0)

        args.n_batch = len(tr_loader)
        args.bs = args.batch_size
        args.num_val_male = len(mal_te_df)
        args.num_val_female = len(fem_te_df)
        # x_tar = x_tr[target_pt]
        tr_info = (tr_loader, x_tar)
        va_info = None
        te_info = test_df
        return tr_info, va_info, te_info


def get_name(args, current_date, fold=0):
    dataset_str = f'{args.dataset}_run_{args.seed}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}'
    model_str = f'{args.mode}_{args.submode}_{args.epochs}_{args.performance_metric}_{args.optimizer}_{args.model_type}_'
    res_str = dataset_str + model_str + date_str
    return res_str

def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]


def get_index_bynot_value(a, val):
    return (a != val).nonzero(as_tuple=True)[0]

def init_history():
    history = {
        'tr_loss': [],
        'tr_acc': [],
        'va_loss': [],
        'va_acc': [],
        'demo_parity': [],
        'acc_parity': [],
        'equal_opp': [],
        'equal_odd': [],
        'te_loss': [],
        'te_acc': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_acc_parity': 0,
        'best_equal_opp': 0,
        'best_equal_odd': 0,
    }
    return history

def save_res(args, dct, name):
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)
