import datetime
import warnings

import torch
import logging
from config import parse_args
from Data.read_data import read_bank, read_abalone
from Utils.utils import seed_everything
from Runs.run_clean import run as run_clean
from Runs.run_fair import run as run_fair

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


def run(args, current_time, device):
    if args.dataset == 'bank':
        train_df, test_df, male_df, female_df, feature_cols, label, z = read_bank(args)
    elif args.dataset == 'abalone':
        train_df, test_df, male_df, female_df, feature_cols, label, z = read_abalone(args)
    fold = 0
    data = (train_df, test_df, male_df, female_df, feature_cols, label, z)

    if args.mode == 'clean':
        run_clean(args, data, current_time, fold, device)
    elif args.mode == 'fairbatch':
        run_fair(args, data, current_time, fold, device)



if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)