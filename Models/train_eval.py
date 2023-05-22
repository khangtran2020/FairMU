import numpy as np
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score
from collections import defaultdict
from torch.autograd import Variable


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        if self.run_mode == 'func' and epoch < self.skip_ep:
            return
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score


class ReduceOnPlatau:
    def __init__(self, mode="max", delta=1e-4, verbose=False, args=None, min_lr=5e-5):
        self.patience = args.lr_patience
        self.counter = 0
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.args = args
        self.min_lr = min_lr
        self.step = args.lr_step
        self.best_score = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                if self.args.lr - self.step < self.min_lr:
                    self.args.lr = self.min_lr
                else:
                    self.args.lr -= self.step
                print("Reduce learning rate to {}".format(self.args.lr))
                self.counter = 0
        else:
            self.best_score = score
            self.counter = 0
        return self.args


def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0

    for bi, d in enumerate(dataloader):
        features, target, _ = d

        features = features.to(device, dtype=torch.float)
        target = torch.squeeze(target).to(device, dtype=torch.float)
        # num_data_point += features.size(dim=0)
        optimizer.zero_grad()
        output = model(features)
        output = torch.squeeze(output)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return train_loss, train_outputs, train_targets


def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss = 0
    # num_data_point = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            # num_data_point += features.size(dim=0)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            loss += loss_eval.item()
            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss, fin_outputs, fin_targets


def demo_parity(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    female_outputs = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, _, _ = d
            features = features.to(device, dtype=torch.float)
            outputs = model(features)
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)

        for bi, d in enumerate(female_loader):
            features, _, _ = d
            features = features.to(device, dtype=torch.float)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)

    male_outputs = np.round(np.array(male_outputs))
    female_outputs = np.round(np.array(female_outputs))
    prob_male = np.sum(male_outputs) / len(male_outputs)
    prob_female = np.sum(female_outputs) / len(female_outputs)
    return prob_male, prob_female, np.abs(prob_male - prob_female)


def equality_of_opp_odd(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    male_target = []
    female_outputs = []
    female_target = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)
            male_target.extend(target.cpu().detach().numpy().astype(int).tolist())

        for bi, d in enumerate(female_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)
            female_target.extend(target.cpu().detach().numpy().astype(int).tolist())

    male_outputs = np.round(np.array(male_outputs))
    tn, fp, fn, tp = confusion_matrix(male_target, male_outputs).ravel()
    male_tpr = tp / (tp + fn)
    male_fpr = fp / (fp + tn)
    female_outputs = np.round(np.array(female_outputs))
    tn, fp, fn, tp = confusion_matrix(female_target, female_outputs).ravel()
    female_tpr = tp / (tp + fn)
    female_fpr = fp / (fp + tn)
    return male_tpr, female_tpr, np.abs(male_tpr - female_tpr), \
        0.5*np.abs(male_fpr - female_fpr)+0.5*np.abs(male_tpr - female_tpr)

def performace_eval(args, y_true, y_pred):
    if args.performance_metric == 'acc':
        return accuracy_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'f1':
        return f1_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif args.performance_metric == 'pre':
        return precision_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
