import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from Data.datasets import Data, FairBatch
from Utils.utils import get_name, save_res
from Models.init import init_model
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Models.train_eval import demo_parity, equality_of_opp_odd


def run(args, data, current_time, fold, device):
    train_df, test_df, male_df, female_df, feature_cols, label, z = data
    df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]]).reset_index(drop=True)
    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]
    val_mal_dataset = Data(df_val_mal[feature_cols].values, df_val_mal[label].values, df_val_mal[z].values)
    val_fem_dataset = Data(df_val_fem[feature_cols].values, df_val_fem[label].values, df_val_fem[z].values)
    valid_dataset = Data(df_valid[feature_cols].values, df_valid[label].values, df_valid[z].values)
    test_dataset = Data(test_df[feature_cols].values, test_df[label].values, test_df[z].values)
    adv_gp_df = male_df[male_df.fold != fold].copy() if len(male_df) > len(female_df) else female_df[
        female_df.fold != fold].copy()
    disadv_gp_df = male_df[male_df.fold != fold].copy() if len(male_df) < len(female_df) else female_df[
        female_df.fold != fold].copy()
    num_data = len(adv_gp_df) + len(disadv_gp_df)
    args.num_feat = len(feature_cols)
    if args.submode == 'clean':
        df_train = pd.concat([adv_gp_df, disadv_gp_df]).reset_index(drop=True)
    elif args.submode == 'sc4':
        adv_gp_pos_df = adv_gp_df[adv_gp_df[label] == 1].copy()
        adv_gp_neg_df = adv_gp_df[adv_gp_df[label] == 0].copy()
        disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[label] == 1].copy()
        disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[label] == 0].copy()
        disadv_gp_pos_df = disadv_gp_pos_df.sample(n=int((1 - args.ratio) * len(disadv_gp_pos_df)), replace=False,
                                                   random_state=args.seed)
        adv_gp_neg_df = adv_gp_neg_df.sample(n=int((1 - args.ratio) * len(adv_gp_neg_df)), replace=False,
                                             random_state=args.seed)
        df_train = pd.concat([adv_gp_pos_df, adv_gp_neg_df, disadv_gp_pos_df, disadv_gp_neg_df])
    elif args.submode == 'sc5':
        adv_gp_pos_df = adv_gp_df[adv_gp_df[label] == 1].copy()
        adv_gp_neg_df = adv_gp_df[adv_gp_df[label] == 0].copy()
        disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[label] == 1].copy()
        disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[label] == 0].copy()
        temp1_df = pd.concat([adv_gp_pos_df, disadv_gp_neg_df], axis=0)
        temp2_df = pd.concat([adv_gp_neg_df, disadv_gp_pos_df], axis=0)
        temp2_df = temp2_df.sample(n=int((1 - args.ratio) * len(temp2_df)), replace=False, random_state=args.seed)
        df_train = pd.concat([temp1_df, temp2_df])
    elif args.submode == 'sc6':
        disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[label] == 1].copy()
        disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[label] == 0].copy()
        num_of_remove = int(args.ratio * len(disadv_gp_df))
        if len(disadv_gp_pos_df) > num_of_remove:
            disadv_gp_pos_df = disadv_gp_pos_df.sample(n=len(disadv_gp_pos_df) - num_of_remove, replace=False,
                                                       random_state=args.seed)
            df_train = pd.concat([adv_gp_df, disadv_gp_pos_df, disadv_gp_neg_df])
        elif len(disadv_gp_pos_df) == num_of_remove:
            df_train = pd.concat([adv_gp_df, disadv_gp_neg_df])
        else:
            disadv_gp_neg_df = disadv_gp_neg_df.sample(n=len(disadv_gp_neg_df) - (num_of_remove - len(disadv_gp_pos_df)),
                                                       replace=False, random_state=args.seed)
            df_train = pd.concat([adv_gp_df, disadv_gp_neg_df])
    elif args.submode == 'sc7':
        num_of_remove = int(args.ratio * len(disadv_gp_df))
        disadv_gp_df = disadv_gp_df.sample(n=len(disadv_gp_df) - num_of_remove, replace=False, random_state=args.seed)
        df_train = pd.concat([adv_gp_df, disadv_gp_df])
    elif args.submode == 'sc8':
        disadv_gp_pos_df = disadv_gp_df[disadv_gp_df[label] == 1].copy()
        disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[label] == 0].copy()
        num_of_remove = int(args.ratio * len(disadv_gp_pos_df))
        disadv_gp_pos_df = disadv_gp_pos_df.sample(n=len(disadv_gp_pos_df) - num_of_remove, replace=False,
                                                   random_state=args.seed)
        df_train = pd.concat([adv_gp_df, disadv_gp_pos_df, disadv_gp_neg_df])
    elif args.submode == 'extreme':
        adv_gp_pos_df = adv_gp_df[adv_gp_df[label] == 1].copy()
        disadv_gp_neg_df = disadv_gp_df[disadv_gp_df[label] == 0].copy()
        df_train = pd.concat([adv_gp_pos_df, disadv_gp_neg_df])
    else:
        df_train = None
    model = init_model(args)
    df_train = df_train.sample(frac=1, replace=False)
    print(f"Before we have {num_data}, After we have {df_train.shape[0]} data points")
    x_train = torch.from_numpy(df_train[feature_cols].values.astype(np.float32)).to(device)
    y_train = torch.from_numpy(df_train[label].values.astype(int)).to(device)
    z_train = torch.from_numpy(df_train[z].values.astype(int)).to(device)
    model.to(device)
    print(f'Length of the remaining set: {len(x_train)}')

    train_dataset = Data(df_train[feature_cols].values, df_train[label].values, df_train[z].values)
    sampler = FairBatch(model, x_train, y_train, torch.tensor(z_train).to(device), batch_size=args.batch_size,
                        alpha=0.005,
                        target_fairness='eqopp', replacement=False, seed=0)
    train_loader = DataLoader(train_dataset, sampler=sampler, num_workers=0)
    val_mal_loader = DataLoader(val_mal_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                pin_memory=True, drop_last=False)
    val_fem_loader = DataLoader(val_fem_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                pin_memory=True, drop_last=False)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                             pin_memory=True, drop_last=False)

    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'demo_parity': [],
        'equality_odd': [],
        'equality_opp': [],
        'best_test': 0
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)

        val_loss, outputs, targets = eval_fn(val_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)

        train_acc = performace_eval(args, train_targets, train_out)
        test_acc = performace_eval(args, test_targets, test_outputs)
        acc_score = performace_eval(args, targets, outputs)

        _, _, demo_p = demo_parity(male_loader=val_mal_loader, female_loader=val_fem_loader,
                                   model=model, device=device)
        _, _, equal_opp, equal_odd = equality_of_opp_odd(male_loader=val_mal_loader,
                                                         female_loader=val_fem_loader, model=model, device=device)

        tk0.set_postfix(Train_Loss=train_loss, Train_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['demo_parity'].append(demo_p)
        history['equality_odd'].append(equal_odd + equal_opp)
        history['equality_opp'].append(equal_opp)
        es(epoch=epoch, epoch_score=acc_score, model=model, model_path=args.save_path + model_name)

    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
    test_acc = performace_eval(args, test_targets, test_outputs)
    history['best_test'] = test_acc
    save_res(name=name, args=args, dct=history)
