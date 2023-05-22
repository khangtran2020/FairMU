import datetime
import warnings
import torch
import logging

from config import parse_args
from Data.read_data import *
from Utils.utils import seed_everything, init_data, get_name
from Runs.run_clean import run as run_clean
from Runs.run_fair import run as run_fair
from Runs.run_attack import run as run_attack

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


def run(args, current_time, device):
    if args.dataset == 'adult': read_dat = read_adult
    elif args.dataset == 'lawschool': read_dat = read_lawschool
    elif args.dataset == 'compas': read_dat = read_compas
    else: read_dat = None

    train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z = read_dat(args)
    args.feature = feature_cols
    args.target = label
    args.z = z
    args.input_dim = len(feature_cols)
    args.output_dim = 1

    print(f'Running with dataset {args.dataset}: {len(train_df)} train, {len(test_df)} test')
    print(f'{len(male_tr_df)} male, {len(female_tr_df)} female, {len(feature_cols)} features')
    print(train_df[args.target].value_counts())
    print(test_df[args.target].value_counts())

    train = (male_tr_df, female_tr_df)
    test = (test_df, male_te_df, female_te_df)
    tr_info, va_info, te_info = init_data(args=args, fold=0, train=train, test=test)
    name = get_name(args=args, current_date=current_time, fold=0)

    run_dict = {
        'clean': run_clean,
        'fair': run_fair,
        'attack': run_attack,
    }

    run = run_dict[args.mode]


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)