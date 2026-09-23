## Modified from OFA repo

# 1K ~ 4K samples per resolution are usually sufficient for training the accuracy predictor.

# In the data preprocessing phase, it is important to make sure the accuracy scale is [0, 1] instead of [0, 100].

# The optimizer is adam. The learning rate is 1e-3. The weight decay is 1e-4. This training setting works well with different batch sizes (e.g., 500, 1000, etc).

# Besides, setting the bias term of the output layer as the average accuracy can improve the training stability.
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class AccPredictorTrainer(object):
    def __init__(self,
                 predictor,
                 train_dataloader,
                 eval_dataloader,
                 save_name,
                 num_epochs,
                 device="cuda:0"):
        self.predictor = predictor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.num_epochs = num_epochs
        self.lr = 1e-3
        self.wd = 1e-4
        self.batch_size = 1000
        
        self.loss_fn = RMSELoss()
        self.device = device
        
        self.save_dir = "models/accuracy_mlp_predictors"
        self.save_name = save_name
        
        
    def train(self, base_acc):
        torch.nn.init.constant_(
            self.predictor.base_acc,
            torch.tensor(base_acc)
        )
        self.optimizer = optim.Adam(self.predictor.parameters(),
                               lr=self.lr,
                               weight_decay=self.wd)
        
        for t in tqdm(range(self.num_epochs)):
            self.train_one_epoch()
            self.evaluate()
        
        
    def train_one_epoch(self):
        model = self.predictor.to(self.device)
        loss_fn = self.loss_fn.to(self.device)
        
        size = len(self.train_dataloader.dataset)
        
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def evaluate(self):
        model = self.predictor.to(self.device)
        loss_fn = self.loss_fn.to(self.device)
        
        model.eval()
        rmse_total = 0
        acc5_total = 0
        acc10_total = 0
        
        size = len(self.eval_dataloader.dataset)
        with torch.no_grad():
             for batch, (X, y) in enumerate(self.eval_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                loss = loss_fn(pred, y)
                # print(pred)
                
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                rmse = np.sqrt(mean_squared_error(pred.cpu().numpy()*100, y.cpu().numpy()*100))
                rmse_total += rmse
                
                acc5 = get_accuracy(pred.cpu().numpy(), y.cpu().numpy(), threshold=0.05)
                acc10 = get_accuracy(pred.cpu().numpy(), y.cpu().numpy(), threshold=0.1)
                acc5_total += acc5
                acc10_total += acc10
        
        rmse_f = rmse_total / len(self.eval_dataloader)
        acc5_f = acc5_total / len(self.eval_dataloader)
        acc10_f = acc10_total / len(self.eval_dataloader)
        # print(f"rmse: {rmse_f:.4f}, acc5: {acc5_f:.4f}, acc10: {acc10_f:.4f}")

    def save(self):
        import os
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        save_name =os.path.join(self.save_dir,self.save_name+'.pth')
        torch.save(self.predictor.state_dict(), save_name)
        print(f'save model in {save_name}')
    
    def load(self, model_path):
        self.predictor.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f'loaded model in {model_path}')

class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

def build_acc_data_loader(
    arch_encoder, codes, accs, batch_size=256, n_workers=16, n_training_sample=None
):
    # load data
    X_all = []
    Y_all = []
    for i in range(len(codes)):
        code = codes[i]
        acc = accs[i]
        X_all.append(arch_encoder.encode(code))
        Y_all.append(acc / 100.0) # range: 0 - 1
    base_acc = np.mean(Y_all)
    # convert to torch tensor
    X_all = torch.tensor(X_all, dtype=torch.float)
    Y_all = torch.tensor(Y_all)

    # random shuffle
    shuffle_idx = torch.randperm(len(X_all))
    X_all = X_all[shuffle_idx]
    Y_all = Y_all[shuffle_idx]

    # split data
    idx = X_all.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
    val_idx = X_all.size(0) // 5 * 4
    X_train, Y_train = X_all[:idx], Y_all[:idx]
    X_test, Y_test = X_all[val_idx:], Y_all[val_idx:]
    print("Train Size: %d," % len(X_train), "Valid Size: %d" % len(X_test))

    # build data loader
    train_dataset = RegDataset(X_train, Y_train)
    val_dataset = RegDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=n_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=n_workers,
    )

    return train_loader, valid_loader, base_acc
