import json


class BlockLUTPredictor(object):
    def __init__(self, lut_path):
        self.block_lut_dict = json.load(open(lut_path))
    
    def predict_efficiency(self, block_dict):
        """
        Return None if existing illegal kernel
        """
        pred = 0
        for block_type, features in block_dict.items():
            
            lut = self.block_lut_dict[block_type]
            for feature in features:
                block_lat = lut[str(feature)]
                
                pred += block_lat
        print(pred)
        return pred