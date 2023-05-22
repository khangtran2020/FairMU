import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from Data.datasets import Data
from sklearn.model_selection import train_test_split


def read_adult(args):
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
              'income']
    label_dict = {
        ' <=50K': '<=50K',
        ' >50K': '>50K',
        ' <=50K.': '<=50K',
        ' >50K.': '>50K'
    }
    train_df = pd.read_csv('Data/Adult/adult.data', header=None)
    test_df = pd.read_csv('Data/Adult/adult.test', skiprows=1, header=None)
    all_data = pd.concat([train_df, test_df], axis=0)
    all_data.columns = header

    def hour_per_week(x):
        if x <= 19:
            return '0'
        elif (x > 19) & (x <= 29):
            return '1'
        elif (x > 29) & (x <= 39):
            return '2'
        elif x > 39:
            return '3'

    def age(x):
        if x <= 24:
            return '0'
        elif (x > 24) & (x <= 34):
            return '1'
        elif (x > 34) & (x <= 44):
            return '2'
        elif (x > 44) & (x <= 54):
            return '3'
        elif (x > 54) & (x <= 64):
            return '4'
        else:
            return '5'

    def country(x):
        if x == ' United-States':
            return 0
        else:
            return 1

    all_data['hours-per-week'] = all_data['hours-per-week'].map(lambda x: hour_per_week(x))
    all_data['age'] = all_data['age'].map(lambda x: age(x))
    all_data['native-country'] = all_data['native-country'].map(lambda x: country(x))
    all_data = all_data.drop(
        ['fnlwgt', 'education-num', 'marital-status', 'occupation', 'relationship', 'capital-gain', 'capital-loss'],
        axis=1)
    temp = pd.get_dummies(all_data['age'], prefix='age')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('age', axis=1)
    temp = pd.get_dummies(all_data['workclass'], prefix='workclass')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('workclass', axis=1)
    temp = pd.get_dummies(all_data['education'], prefix='education')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('education', axis=1)
    temp = pd.get_dummies(all_data['race'], prefix='race')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('race', axis=1)
    temp = pd.get_dummies(all_data['hours-per-week'], prefix='hour')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('hours-per-week', axis=1)
    all_data['income'] = all_data['income'].map(label_dict)
    lb = LabelEncoder()
    all_data['sex'] = lb.fit_transform(all_data['sex'].values)
    lb = LabelEncoder()
    all_data['income'] = lb.fit_transform(all_data['income'].values)
    feature_cols = list(all_data.columns)
    feature_cols.remove('income')
    feature_cols.remove('sex')
    label = 'income'
    z = 'sex'
    if args.mode == 'func':
        all_data = minmax_scale(df=all_data, cols=feature_cols)
        all_data['bias'] = 1.0
        feature_cols.append('bias')
    train_df = all_data[:train_df.shape[0]].reset_index(drop=True)
    test_df = all_data[train_df.shape[0]:].reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
        train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    else:
        train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z

def read_compas(args):
    data_dir = 'Data/Compas/compas-scores-two-years.csv'
    df = pd.read_csv(data_dir)
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != '0']
    df = df[df['score_text'] != 'N/A']

    df['in_custody'] = pd.to_datetime(df['in_custody'])
    df['out_custody'] = pd.to_datetime(df['out_custody'])
    df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()
    df.drop(
        [
            'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
            'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
        ], axis=1, inplace=True
    )
    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    df['two_year_recid'] = 1 - df['two_year_recid']
    label = 'two_year_recid'
    z = 'race'
    features = ['age', 'sex', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree', 'v_score_text']
    df = df[features + [label, z]]
    categorical_columns = []
    for col in features:
        print(col, df[col].dtype)
        if df[col].isna().sum() > 0:
            df = df.drop(col, axis=1)
            features.remove(col)
        else:
            if df[col].dtype == object:
                categorical_columns += [col]
            else:
                mean, std = df[col].mean(), df[col].std()
                df[col] = (df[col] - mean) / std

    df[z] = df[z].map({
        'African-American': 0,
        'Caucasian': 1
    })
    df = pd.get_dummies(df, columns=categorical_columns, prefix_sep='_')
    feature_cols = list(df.columns)
    feature_cols.remove(label)
    feature_cols.remove(z)

    train_df, test_df, _, _ = train_test_split(df, df[label], test_size=0.2, stratify=df[label])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
        train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    else:
        train_df = pd.concat([male_tr_df, female_tr_df], axis=0).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z


def read_lawschool(args):
    df = pd.read_stata('Data/Lawschool/lawschs1_1.dta')
    df.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True, subset=['admit'])
    df.replace(to_replace='', value=np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    df = df[df['race'] != 'Asian']
    feature_cols = list(df.columns)
    feature_cols.remove('gender')
    categorical_columns = []
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df.drop(col, axis=1, inplace=True)
        else:
            if df[col].dtype == np.object:
                categorical_columns.append(col)
            else:
                df[col] = (df[col] - df[col].mean())/(df[col].std() + 1e-12)
    df['race'] = df['race'].apply(lambda x: int(x != 'White'))
    categorical_columns.remove('race')
    df = pd.get_dummies(df, columns=categorical_columns, prefix_sep='=')
    feature_cols = list(df.columns)
    feature_cols.remove('admit')
    feature_cols.remove('race')
    label = 'admit'
    z = 'race'
    df[label] = df[label].values.astype(int)
    df[z] = df[z].values.astype(int)
    if args.mode == 'func':
        df = minmax_scale(df=df, cols=feature_cols)
        df['bias'] = 1.0
        feature_cols.append('bias')
    train_df, test_df, _, _ = train_test_split(df, df[label], test_size=0.2, stratify=df[label])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
        train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    else:
        train_df = pd.concat([male_tr_df, female_tr_df], axis=0).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z

def fold_separation(train_df, folds, feat_cols, label):
    skf = StratifiedKFold(n_splits=folds)
    train_df['fold'] = np.zeros(train_df.shape[0])
    for i, (idxT, idxV) in enumerate(skf.split(train_df[feat_cols], train_df[label])):
        train_df['fold'][idxV] = i


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


