import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from Data.datasets import Data
from sklearn.model_selection import train_test_split

def read_bank(args):
    # 3305
    df = pd.read_csv('Data/Bank/formated_bank.csv')
    feature_cols = list(df.columns)
    feature_cols.remove('y')
    feature_cols.remove('z')
    feature_cols.remove('label')
    feature_cols.remove('is_train')
    feature_cols.remove('intercept')
    label = 'y'
    z = 'z'
    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)
    male_df = train_df[train_df['z'] == 1].copy().reset_index(drop=True)
    female_df = train_df[train_df['z'] == 0].copy().reset_index(drop=True)

    fold_separation(male_df, args.folds, feature_cols, label)
    fold_separation(female_df, args.folds, feature_cols, label)
    train_df = pd.concat([male_df, female_df], axis=0).reset_index(drop=True)
    return train_df, test_df, male_df, female_df, feature_cols, label, z

def read_abalone(args):
    # 1436
    df = pd.read_csv('Data/Abalone/formated_abalone.csv')
    # print(df.head())
    feature_cols = list(df.columns)
    feature_cols.remove('y')
    feature_cols.remove('label')
    feature_cols.remove('z')
    feature_cols.remove('is_train')
    label = 'y'
    z = 'z'
    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)
    male_df = train_df[train_df['z'] == 1].copy().reset_index(drop=True)
    female_df = train_df[train_df['z'] == 0].copy().reset_index(drop=True)
    fold_separation(male_df, args.folds, feature_cols, label)
    fold_separation(female_df, args.folds, feature_cols, label)
    train_df = pd.concat([male_df, female_df], axis=0).reset_index(drop=True)
    return train_df, test_df, male_df, female_df, feature_cols, label, z

def fold_separation(train_df, folds, feat_cols, label):
    skf = StratifiedKFold(n_splits=folds)
    train_df['fold'] = np.zeros(train_df.shape[0])
    for i, (idxT, idxV) in enumerate(skf.split(train_df[feat_cols], train_df[label])):
        train_df.at[idxV, 'fold'] = i

def minmax_scale(df, cols):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for col in cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df

def choose_data(args, df_0, df_1):
    # print(len(df_0),len(df_1))
    if len(df_0) > len(df_1):
        df = df_1.copy()
        df_1 = df_0.copy()
        df_0 = df.copy()
        del (df)

    df_0 = df_0.reset_index(drop=True)
    df_1 = df_1.reset_index(drop=True)

    num_pt = int(args.ratio * len(df_0))
    if num_pt > len(df_1):
        print('Can not achieve that rate between group0 and group1')
        args.can_ratio = False
        return df_0.reset_index(drop=True), df_1.reset_index(drop=True)
    else:
        idx = np.random.choice(np.arange(len(df_1)), size=num_pt, replace=False)
        df_1 = df_1.iloc[idx, :].copy()
        return df_0.reset_index(drop=True), df_1.reset_index(drop=True)

def get_UTK(args):
    utk_data_path = "Data/UTK/age_gender.gz"
    label = 'gender'
    z = 'ethnicity'
    df = pd.read_csv(utk_data_path, compression='gzip')
    X = df.pixels.apply(lambda x: np.array(x.split(" "), dtype=float))
    X = np.stack(X)
    X = X / 255.0
    X = X.astype('float32').reshape(X.shape[0], 1, 48, 48)
    y = df[label]
    z = df[z]
    np.random.seed(0)  # random seed of partition data into train/test
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.16)
    train_dataset = Data(X=x_train, y=y_train.values, ismale=z.iloc[y_train.index].values)
    valid_dataset = Data(X=x_valid, y=y_valid.values, ismale=z.iloc[y_valid.index].values)
    test_dataset = Data(X=x_test, y=y_test.values, ismale=z.iloc[y_test.index].values)
    num_group = z.nunique()
    z_train = z.iloc[y_train.index]
    z_test = z.iloc[y_test.index]
    groups_train_dataset = {}
    groups_test_dataset = {}
    args.num_group = num_group
    group_num = {}
    for i in range(num_group):
        groups_train_dataset['group_{}'.format(i)] = Data(
            X=x_train[list(z_train[z_train == i].reset_index(drop=True).index)],
            y=y_train.iloc[z_train[z_train == i].reset_index(drop=True).index].values,
            ismale=z_train[z_train == i].values)
        groups_test_dataset['group_{}'.format(i)] = Data(
            X=x_test[list(z_test[z_test == i].reset_index(drop=True).index)],
            y=y_test.iloc[z_test[z_test == i].reset_index(drop=True).index].values,
            ismale=z_test[z_test == i].values)
        group_num['group_{}'.format(i)] = len(groups_test_dataset['group_{}'.format(i)])
    return train_dataset, valid_dataset, test_dataset, groups_train_dataset, groups_test_dataset, group_num
