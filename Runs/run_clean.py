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

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(tr_loader, model, criterion, optimizer, device,
                                                        scheduler=None)

        val_loss, outputs, targets = eval_fn(va_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(te_loader, model, criterion, device)

        train_acc = performace_eval(args, train_targets, train_out)
        test_acc = performace_eval(args, test_targets, test_outputs)
        acc_score = performace_eval(args, targets, outputs)

        _, _, demo_p = demo_parity(male_loader=te_mal_loader, female_loader=te_fem_loader,
                                   model=model, device=device)
        _, _, equal_opp, equal_odd = equality_of_opp_odd(male_loader=te_mal_loader,
                                          female_loader=te_fem_loader, model=model, device=device)


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
    test_loss, test_outputs, test_targets = eval_fn(te_loader, model, criterion, device)
    test_acc = performace_eval(args, test_targets, test_outputs)
    history['best_test'] = test_acc
    save_res(name=name, args=args, dct=history)
