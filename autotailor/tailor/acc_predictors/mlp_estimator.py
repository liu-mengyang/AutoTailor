import torch

class MLPEstimator:
    def __init__(self, arch_encoder, mlp_predictor, device="cuda:0"):
        self.device=device
        self.arch_encoder = arch_encoder
        self.predictor = mlp_predictor
    
    def predict_accuracy(self, sample):
        model = self.predictor.to(self.device)
        self.predictor.eval()
        X = self.arch_encoder.encode(sample)
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        with torch.no_grad():
            pred = self.predictor(X)
            # print(pred)
            return pred.cpu().numpy()*100
        
    def predict_efficiency(self, sample):
        model = self.predictor.to(self.device)
        self.predictor.eval()
        X = self.arch_encoder.encode(sample)
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        with torch.no_grad():
            pred = self.predictor(X)
            # print(pred)
            return pred.cpu().numpy()*100