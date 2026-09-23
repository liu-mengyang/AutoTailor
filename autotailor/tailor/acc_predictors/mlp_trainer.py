## Modified from OFA repo

# 1K ~ 4K samples per resolution are usually sufficient for training the accuracy predictor.

# In the data preprocessing phase, it is important to make sure the accuracy scale is [0, 1] instead of [0, 100].

# The optimizer is adam. The learning rate is 1e-3. The weight decay is 1e-4. This training setting works well with different batch sizes (e.g., 500, 1000, etc).

# Besides, setting the bias term of the output layer as the average accuracy can improve the training stability.
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .evaluation_data import SCHEMA, dataset_identity


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
        self.split_metadata = getattr(train_dataloader, "split_metadata", None)
        if train_dataloader is not None and train_dataloader is eval_dataloader:
            raise ValueError("Training loader cannot also be the held-out evaluation loader")
        
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
            if self.eval_dataloader is not None:
                self.evaluate()
        
        
    def train_one_epoch(self):
        model = self.predictor.to(self.device)
        model.train()
        loss_fn = self.loss_fn.to(self.device)
        
        size = len(self.train_dataloader.dataset)
        
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X).reshape(-1)
            loss = loss_fn(pred, y.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def evaluate(self):
        if self.eval_dataloader is None or not len(self.eval_dataloader.dataset):
            raise ValueError("No held-out data available for evaluation")
        model = self.predictor.to(self.device)
        was_training = model.training
        model.eval()
        squared_error = 0.0
        acc5_count = acc10_count = count = 0
        try:
            with torch.no_grad():
                for X, y in self.eval_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X).reshape(-1)
                    y = y.reshape(-1)
                    error = pred - y
                    squared_error += error.double().square().sum().item()
                    acc5_count += (error.abs() <= 0.05 * y.abs()).sum().item()
                    acc10_count += (error.abs() <= 0.1 * y.abs()).sum().item()
                    count += y.numel()
        finally:
            model.train(was_training)
        return {
            "n": count,
            "rmse_pp": (squared_error / count) ** 0.5 * 100.0,
            "acc5": acc5_count / count,
            "acc10": acc10_count / count,
        }

    def save(self):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        save_name = os.path.join(self.save_dir, self.save_name + '.pth')
        torch.save({
            "state_dict": self.predictor.state_dict(),
            "split_metadata": self.split_metadata,
        }, save_name)
        self.predictor.split_metadata = self.split_metadata
        print(f'save model in {save_name}')
        return save_name

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.split_metadata = checkpoint.get("split_metadata")
        self.predictor.split_metadata = self.split_metadata
        self.predictor.load_state_dict(checkpoint.get("state_dict", checkpoint))
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
    arch_encoder, codes, accs, batch_size=256, n_workers=16, n_training_sample=None,
    seed=0,
):
    features, code_ids, feature_ids, dataset_id = dataset_identity(arch_encoder, codes, accs)
    # Reject duplicate architectures, including different codes with equal encodings.
    # Splitting individual duplicate rows would let the same architecture cross sets.
    if len(set(code_ids)) != len(codes) or len(set(feature_ids)) != len(codes):
        raise ValueError("Duplicate architectures; canonicalize and deduplicate before splitting")
    size = len(codes)
    idx = size // 5 * 4 if n_training_sample is None else n_training_sample
    if not isinstance(idx, int) or not 1 <= idx <= size:
        raise ValueError("n_training_sample must be between 1 and the dataset size")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    train_indices, validation_indices = indices[:idx], indices[idx:]
    X_all = torch.tensor(features, dtype=torch.float32)
    Y_all = torch.tensor(accs, dtype=torch.float32) / 100.0
    train_dataset = RegDataset(X_all[train_indices], Y_all[train_indices])
    base_acc = float(Y_all[train_indices].mean().item())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader.split_metadata = {
        "schema": SCHEMA,
        "seed": seed,
        "dataset_sha256": dataset_id,
        "training_indices": train_indices,
        "validation_indices": validation_indices,
        "training_architecture_sha256": [code_ids[i] for i in train_indices],
        "training_feature_sha256": [feature_ids[i] for i in train_indices],
        "base_accuracy_fraction": base_acc,
        "holdout_scope": "architecture only; image split requires collection provenance",
    }
    valid_loader = None
    if validation_indices:
        valid_loader = torch.utils.data.DataLoader(
            RegDataset(X_all[validation_indices], Y_all[validation_indices]),
            batch_size=batch_size, shuffle=False, num_workers=n_workers,
        )
    print(f"Train Size: {idx}, Valid Size: {len(validation_indices)}")
    return train_loader, valid_loader, base_acc
