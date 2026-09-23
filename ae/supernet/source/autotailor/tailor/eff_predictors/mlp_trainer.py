import os

from loguru import logger
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as Data
import torchmetrics
from tqdm import tqdm

from .mlp_predictor import MLP


class Trainer:
    def __init__(
       self,
       train_dataset=None,
       eval_dataset=None,
       kernel_type=None,
       batch_size=128,
       learning_rate=0.001,
       epochs=350,
       save_dir = "models/latency_mlp_predictors",
       weights = None,
       output_name = None,
   ):   
        self.train_dataset = train_dataset 
        self.eval_dataset = eval_dataset
        self.kernel_type = kernel_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_fn = None
        self.save_dir = save_dir
        self.output_name = output_name
        
        self.model = MLP(input_features=len(train_dataset[0][0]))
        if weights is not None:
            self.model.load_state_dict(weights)
            print('successfully load similar device weights!')
        
    def create_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
    
    def create_loss_fn(self, loss_fn=None):
        if loss_fn == None:
            self.loss_fn = torchmetrics.MeanAbsolutePercentageError()
        else:
            self.loss_fn = loss_fn
       
    def train_one_epoch(self, model, dataloader, loss_fn):
        size = len(dataloader.dataset)
        
        for batch, (X, y) in enumerate(dataloader):
            if torch.cuda.is_available():
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
    
    def train(self, sample_num=-1, train_dataloader=None,):
        if torch.cuda.is_available():
            model = self.model.cuda()
        else:
            model = self.model
        model.train()
        
        epochs = self.epochs
        train_dataloader = self.get_dataloader(self.train_dataset)

        if self.loss_fn == None:
            self.create_loss_fn()
        if torch.cuda.is_available():
            loss_fn = self.loss_fn.cuda()
        else:
            loss_fn = self.loss_fn
        self.create_optimizer_and_scheduler()

        for t in tqdm(range(epochs)):
            self.train_one_epoch(model, train_dataloader,loss_fn)
            
        if self.eval_dataset:
            self.evaluate(sample_num)

    def save(self):
        import os
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        savename = os.path.join(self.save_dir, self.output_name+'.pth')
        torch.save(self.model.state_dict(), savename)
        logger.info(f'save model in {savename}')
        return savename


    def evaluate(self, sample_num):
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
 
    def get_dataloader(self, dataset):
        dataLoader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return dataLoader


def split_train_test(X, Y):
    assert(len(X) == len(Y))

    if len(X) == 1:
        trainx, testx, trainy, testy = X, X, Y, Y
    else:
        trainx, testx, trainy, testy = train_test_split(X, Y, test_size = 0.2, random_state = 10)
    return trainx, testx, trainy, testy


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


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)