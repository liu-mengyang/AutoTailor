from tensorflow import keras

from ..interface import BaseOperator

class GEMM(BaseOperator):
    def get_model(self):
        cin = self.config['CIN']
        cout = self.config['COUT']
        hw = self.config['HW']
        return keras.layers.Dense(
            cout,
            use_bias=False
        )

    def get_output_shape(self):
        cout = self.config["COUT"]
        output_hw = self.config['HW']
        return [output_hw, cout]

class Conv(BaseOperator):
    def get_model(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        return keras.layers.Conv2D(
            cout,
            kernel_size=self.config["KERNEL_SIZE"],
            strides=self.config["STRIDES"],
            padding="same"
        )

    def get_output_shape(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        output_h = (self.input_shape[0] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        return [output_h, output_w, cout]


class DepthConv(BaseOperator):
    def get_model(self):
        return keras.layers.DepthwiseConv2D(
            kernel_size=self.config["KERNEL_SIZE"],
            strides=self.config["STRIDES"],
            padding="same"
        )
        
    def get_output_shape(self):
        cs = self.input_shape[2] if "CHANNEL_SIZE" not in self.config else self.config["CHANNEL_SIZE"]
        output_h = (self.input_shape[0] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        return [output_h, output_w, cs]
        
        

