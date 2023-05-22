import torch
from Utils.utils import save_res, init_history
from Models.init import init_model
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Models.train_eval import demo_parity, equality_of_opp_odd

def run(args, tr_info, va_info, te_info, name, device):
    model_name = '{}.pt'.format(name)
    tr_loader, _, _ = tr_info
    va_loader = va_info
    te_loader, te_mal_loader, te_fem_loader = te_info

    model = init_model(args)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = init_history()