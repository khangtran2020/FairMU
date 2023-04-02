from Models.models import NeuralNetwork, Logit
from torch.optim import Adam, AdamW, SGD


def init_model(args):
    print("Training with graph {}".format(args.model_type))
    model = None
    if args.model_type == 'nn':
        model = NeuralNetwork(input_dim=args.num_feat, hidden_dim=args.hid_dim, output_dim=1, n_layers=args.n_hid)
    elif args.model_type == 'lr':
        model = Logit(input_dim=args.num_feat, hidden_dim=args.hid_dim, output_dim=1)
    return model


def init_optimizer(optimizer_name, model, lr):
    print("Optimizing with optimizer {}".format(optimizer_name))
    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr)
    return optimizer