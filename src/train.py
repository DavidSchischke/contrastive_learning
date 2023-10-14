import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from torchmetrics import F1Score, Accuracy


def train_epoch(model: Module, train_data: DataLoader, loss, optim: Optimizer, device: str = "cpu"): 
    """
    Train model for one epoch

    Args:
        model (Module): Model
        train_data (DataLoader): Training Data
        loss (_type_): Loss func
        optim (Optimizer): Optimizer
        device (str, optional): Training Device. Defaults to "cpu".
    """
    model.to(device)
    
    model.train()
    for X, y in train_data: 
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        
        optim.zero_grad()
        l = loss(y_hat, y)
        l.backward()
        optim.step()

def eval_epoch(model: Module, test_data: DataLoader, n_classes: int = 47, device: str = "cpu"):
    y_hat_all = []
    y_true_all = []

    acc = Accuracy(task = "multiclass", num_classes = n_classes).to(device)
    f1 = F1Score(task = "multiclass", num_classes = n_classes).to(device)
    
    model.eval()
    with torch.no_grad():
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            
            y_hat_all.append(y_hat)
            y_true_all.append(y)
    
    y_hat_all = torch.cat(y_hat_all).to(device)
    y_true_all = torch.cat(y_true_all).to(device)

    print(f"Test Accuracy: {acc(y_hat_all, y_true_all):>.4f}, Test F1: {f1(y_hat_all, y_true_all):>.4f}")
        

if __name__ == "__main__": 
    import sys 
    sys.path.append(".")
    
    from torch.nn import CrossEntropyLoss
    from torch.optim import SGD
    
    from src.utils import get_device
    from src.model import ExampleNet
    from src.dataloader import load_emnist

    device = get_device()

    model = ExampleNet()
    train_data, test_data = load_emnist()

    loss = CrossEntropyLoss()
    optim = SGD(model.parameters(), 
                lr = 0.03, 
                momentum = 0.9,
                weight_decay=1e-3)

    for epoch in range(10): 
        train_epoch(model, train_data, loss, optim, device)
        eval_epoch(model, test_data, device = device)