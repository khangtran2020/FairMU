import torch
import numpy as np
from Utils.utils import save_res, init_history
from Models.init import init_model
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Models.train_eval import demo_parity, equality_of_opp_odd
from Utils.fairrr import fairRR
from sklearn.metrics import accuracy_score

def run(args, tr_info, va_info, te_info, name, device):
    model_name = '{}.pt'.format(name)
    tr_loader, x_tar = tr_info
    va_loader = va_info
    te_df = te_info

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

    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(tr_loader, model, criterion, optimizer, device,
                                                        scheduler=None, mode='attack')

        # val_loss, outputs, targets = eval_fn(va_loader, model, criterion, device)
        # test_loss, test_outputs, test_targets = eval_fn(te_loader, model, criterion, device)

        train_acc = performace_eval(args, train_targets, train_out)
        # test_acc = performace_eval(args, test_targets, test_outputs)
        # acc_score = performace_eval(args, targets, outputs)

        # _, _, demo_p = demo_parity(male_loader=te_mal_loader, female_loader=te_fem_loader,
        #                            model=model, device=device)
        # _, _, equal_opp, equal_odd = equality_of_opp_odd(male_loader=te_mal_loader,
        #                                                  female_loader=te_fem_loader, model=model, device=device)
        tk0.set_postfix(Train_Loss=train_loss, Train_SCORE=train_acc)

        history['tr_loss'].append(train_loss)
        history['tr_acc'].append(train_acc)
        # history['val_history_loss'].append(val_loss)
        # history['val_history_acc'].append(acc_score)
        # history['test_history_loss'].append(test_loss)
        # history['test_history_acc'].append(test_acc)
        # history['demo_parity'].append(demo_p)
        # history['equality_odd'].append(equal_odd + equal_opp)
        # history['equality_opp'].append(equal_opp)
        es(epoch=epoch, epoch_score=train_acc, model=model, model_path=args.save_path + model_name)
        # if train_acc > 0.9: break

    model.load_state_dict(torch.load(args.save_path + model_name))
    # testing time
    x_te = te_df[args.feature].values
    x_te = np.delete(x_te, args.tar_pt, 0)
    print(x_te.shape)
    if args.subsubmode == 'remove':
        x_tar_temp = np.stack([x_tar for _ in range(x_te.shape[0])])
        dist = np.linalg.norm(x_tar_temp - x_te, ord=2, axis=1)
        result = np.argpartition(dist, args.top_k)
        x_te = np.delete(x_te, result[:args.top_k], 0)
        print(x_te.shape)

    if args.submode == 'def':
        x_te = fairRR(arr=x_te, eps=args.tar_eps, num_int=1, num_bit=8, mode='relax')

    label = []
    prediction = []
    with torch.no_grad():
        print("Prediction on target:", model(torch.from_numpy(np.expand_dims(x_tar, axis=0).astype(np.float32))))
        threshold = model(torch.from_numpy(np.expand_dims(x_tar, axis=0).astype(np.float32))).item() - 1e-12
        # print(threshold - 5e-4)
        for i in range(100):
            test_arr_idx = np.random.choice(a=np.arange(x_te.shape[0]), size=5, replace=False)
            test_arr = x_te[test_arr_idx]
            sample = np.random.uniform(0, 1, 1)[0]
            if sample > 0.5:
                label.append(1)
                test_arr = np.concatenate((test_arr, np.expand_dims(x_tar, axis=0)), axis=0)
            else:
                label.append(0)
            test_arr = torch.from_numpy(test_arr.astype(np.float32))
            output = model(test_arr)
            output = np.round(torch.squeeze(output).cpu().detach().numpy()).astype(int)
            if np.sum(output) > 0.0:
                prediction.append(1)
                print(f"label {sample > 0.5}, prediction {1}")
            else:
                prediction.append(0)
                print(f"label {sample > 0.5}, prediction {0}")
    print("Attack performance:", accuracy_score(y_true=label,y_pred=prediction))