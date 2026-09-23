import tensorflow.keras as keras

from .utils import get_inputs_by_shapes
from .operators import *
from ..interface import BaseBlock


class TFBlock(BaseBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        if 'KERNEL_SIZE' not in config:   
            #gemm
            self.input_shape = [config['HW'],config['CIN']]
            self.input_tensor_shape = [self.input_shape]
        else:
            self.input_shape = [config["HW"], config["HW"], config["CIN"] if "CIN" in config else config["CHANNEL_SIZE"]]
            self.input_tensor_shape = [self.input_shape]
        self.batch_size = batch_size
        
    def test_block(self):
        raise NotImplementedError
        import os, shutil
        from typing import List
        model_path = "./temp_model"
        model = self.get_model()
        model_output = model(get_inputs_by_shapes(self.input_tensor_shape))
        
        # check model save and reload
        keras.models.save_model(model, model_path)
        restore_model = keras.models.load_model(model_path)
        if isinstance(model_output, List):
            output_shape = [mod.shape for mod in model_output]
            restore_output_shape = [mod.shape for mod in restore_model(get_inputs_by_shapes(self.input_tensor_shape))]
        else:
            output_shape = model_output.shape
            restore_output_shape = restore_model(get_inputs_by_shapes(self.input_tensor_shape)).shape
        assert output_shape == restore_output_shape
        shutil.rmtree(model_path)

        # check model convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(restore_model)
        tflite_model = converter.convert()
        open(model_path + '.tflite', 'wb').write(tflite_model)
        os.remove(model_path + '.tflite')
        logging.keyinfo("Testing block is success!")

    def save_model(self, save_path):
        model = self.get_model()
        keras.models.save_model(model, save_path)

    def build_model(self, ops):
        ''' convert a list of operators to keras model.
        '''
        class Model(keras.Model):
            def __init__(self, ops):
                super().__init__()
                self.ops = keras.Sequential(ops)

            def call(self, inputs):
                print(inputs)
                x = self.ops(inputs)
                return x

        model = Model(ops)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model

    def get_model(self):
        raise NotImplementedError

class GEMMBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        gemm_op = GEMM(self.input_shape, config)
        self.gemm_op = gemm_op.get_model()

    def get_model(self):
        return self.build_model([self.gemm_op])
    
class ConvBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op = conv_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op])


class DwConvBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DepthConv(self.input_shape, config)
        self.dwconv_op = dwconv_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op])