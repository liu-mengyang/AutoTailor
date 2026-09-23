

class FLOPsCounter(object):
    def __init__(self, tailor):
        self.tailor = tailor
    
    def predict_accuracy(self, code):
        self.tailor.transform(code)
        flops, params = self.tailor.tir.count_flops_params()
        return flops