import torch
import torchmetrics
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from loguru import logger
import numpy as np


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)

def latency_metrics(y_pred, y_true):
    """
    evaluation metrics for prediction performance
    """
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15


class LitePredMLP(nn.Module):
    def __init__(self,input_features=None):
        super().__init__()
        self.fc1 = nn.Linear(input_features,16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 128)
        self.fc14 = nn.Linear(128, 64)
        self.fc15 = nn.Linear(64, 16)
        self.fc16 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.leakyrelu(self.fc13(x))
        x = self.leakyrelu(self.fc14(x))
        x = self.relu(self.fc15(x))
        out = self.fc16(x)

        return out.squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, latent_dims=2, hidden_dims=8, output_dims=6, layers=2):
        super(Decoder, self).__init__()
        self.linear_in = nn.Linear(latent_dims, hidden_dims)
        mid_layers = [nn.Linear(hidden_dims, hidden_dims) for _ in range(layers - 2)]
        self.linear_mid = nn.ModuleList(mid_layers)
        self.linear_out = nn.Linear(hidden_dims, output_dims)

    def forward(self, z):
        z = self.linear_in(z)
        z = F.relu(z)
        for linear_layer in self.linear_mid:
            z = F.relu(z)
            z = linear_layer(z)
        z = self.linear_out(z)
        z = torch.sigmoid(z)
        return z


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=2, hidden_dims=8, input_dims=6, layers=2):
        super(VariationalEncoder, self).__init__()
        self.linear_in = nn.Linear(input_dims, hidden_dims)
        mid_layers = [nn.Linear(hidden_dims, hidden_dims) for _ in range(layers - 2)]
        self.linear_mid = nn.ModuleList(mid_layers)
        self.linear_mu = nn.Linear(hidden_dims, latent_dims)
        self.linear_sigma = nn.Linear(hidden_dims, latent_dims)
        self.kl = 0

    def forward(self, x):
        x = self.linear_in(x)
        x = F.relu(x)
        for linear_layer in self.linear_mid:
            x = linear_layer(x)
            x = F.relu(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z
    

class VariationalAutoencoder(nn.Module):
    def __init__(
            self, latent_dims=2, source_dims=6,
            encoder_hidden_dims=256, decoder_hidden_dims=256,
            encoder_layers=2, decoder_layers=3
        ):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, encoder_hidden_dims, source_dims, encoder_layers)
        self.decoder = Decoder(latent_dims, decoder_hidden_dims, source_dims, decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LitePredTrainer:
    def __init__(
       self,
       features,
       train_dataset=None,
       eval_dataset=None,
       kernel_type=None,
       batch_size=32,
       learning_rate=0.001,
       epochs=350,
       save_dir = "models/latency_litpred_predictors",
       weights = None,
       output_name = None
   ):   
        self.train_dataset = train_dataset 
        self.eval_dataset = eval_dataset
        self.kernel_type = kernel_type if kernel_type else 'kernel'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_fn = None
        self.save_dir = save_dir if save_dir else '../predictors'
        self.output_name = output_name
        
        self.model = LitePredMLP(input_features=features[kernel_type])
        if weights is not None:
            self.model.load_state_dict(weights)
            print('successfully load similar device weights!')
        
    
    def create_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
    def create_loss_fn(self,loss_fn=None):
        if loss_fn == None:
            self.loss_fn = torchmetrics.MeanAbsolutePercentageError()
        else:
            self.loss_fn = loss_fn
       
    def train_one_epoch(self,model,dataloader,loss_fn):
        size = len(dataloader.dataset)
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            loss = loss_fn(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        self.scheduler.step()
        
        
    def train(self):
        model = self.model.cuda()
        model.train()
        
        epochs = self.epochs
        train_dataloader = self.get_dataloader(self.train_dataset)

        if self.loss_fn == None:
            self.create_loss_fn()
        loss_fn = self.loss_fn.cuda()
        self.create_optimizer_and_scheduler()

        for t in tqdm(range(epochs)):
            self.train_one_epoch(model, train_dataloader,loss_fn)
            
        if self.eval_dataset:
            self.evaluate()

    def save(self):
        import os
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        savename = os.path.join(self.save_dir,self.output_name+'.pth')
        torch.save(self.model.state_dict(), savename)
        print(f'save model in {savename}')
    
    def evaluate(self):
        dataloader = self.get_dataloader(self.eval_dataset)
        
        if self.loss_fn == None:
            self.create_loss_fn()
        if torch.cuda.is_available():
            loss_fn = self.loss_fn.cuda()
        else:
            loss_fn = self.loss_fn
        num_batches = len(dataloader)
        if torch.cuda.is_available():
            model = self.model.cuda()
        else:
            model = self.model
        model.eval()
        test_loss = 0
        rmse_total, rmspe_total, error_total, acc5_total, acc10_total, acc15_total = 0,0,0,0,0,0
        with torch.no_grad():
            for X, y in tqdm(dataloader):
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                y1= y.cpu().numpy()
                pred1 = pred.cpu().numpy()
                rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(pred1, y1)
                rmse_total += rmse
                rmspe_total += rmspe
                error_total += error
                acc5_total += acc5
                acc10_total += acc10
                acc15_total += acc15
        test_loss /= num_batches
        rmse_f = rmse_total / num_batches
        rmspe_f = rmspe_total / num_batches
        error_f = error_total / num_batches
        acc5_f = acc5_total / num_batches
        acc10_f = acc10_total / num_batches
        acc15_f = acc15_total / num_batches
        logger.info(f"mlp: rmse: {rmse_f:.4f}; rmspe: {rmspe_f:.4f}; error: {error_f:.4f}; 5% accuracy: {acc5_f:.4f}; 10% accuracy: {acc10_f:.4f}; 15% accuracy: {acc15_f:.4f}.")
        logger.info(f"Test Error: \n 10% Accuracy: {acc10_f:.4f}, Avg loss: {test_loss:>8f} \n")
        
        
    def get_dataloader(self,dataset):
        dataLoader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return dataLoader