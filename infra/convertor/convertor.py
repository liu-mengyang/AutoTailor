import copy
import os
import time
import subprocess
import shutil
import warnings

from loguru import logger
import onnx
import onnxoptimizer as oop
from onnxconverter_common import float16
import torch
import onnx2torch
import onnx2tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from torch.utils.mobile_optimizer import optimize_for_mobile

from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import get_default_qconfig


def timestamp(name, stage):
    logger.info('TIMESTAMP, %s, %s, %f' % (name, stage, time.time()))
    # print('TIMESTAMP, %s, %s, %f' % (name, stage, time.time()), file=sys.stderr)


class Convertor:
    """Model format converter."""
    
    def __init__(self):
        self.support_backend = ['tflite', 'ncnn', 'onnx', 'pytorch']
        self.save_root = os.getenv('MODEL_HOME')

    def torch2pytorch(self, model, model_name, data_shape, save_dir=None, enable_int8=False, verbose=False):
        # check disk
        usage_ratio = self.check_disk()
        if usage_ratio >= 95:
            raise OSError('Disk free space is not enough.')
        elif usage_ratio >= 80:
            warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        if save_dir is None:
            save_dir = self.save_root+"/pytorch"
        
        trace_model_path = os.path.join(save_dir, (model_name+'.pt'))

        model = model.eval()
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(data_shape, dict):
            x = []
            str_shape = ''
            for k, v in data_shape.items():
                x.append(torch.rand(v))
                if len(str_shape) == 0:
                    str_shape+=(str(v).replace(' ', ''))
                else:
                    str_shape+=(','+str(v).replace(' ', ''))
            
        else:
            x = torch.rand(data_shape)
            str_shape = str(data_shape).replace(' ', '')
        if enable_int8:
            qconfig = get_default_qconfig("qnnpack")
            qconfig_dict = {
                "": qconfig,
            }
            model_to_quantize = copy.deepcopy(model)
            prepared_model = prepare_fx(model_to_quantize, qconfig_dict, x)
            quantized_model = convert_fx(prepared_model)            
            scripted_quantized_model = torch.jit.script(quantized_model)
            torch.jit.save(scripted_quantized_model, trace_model_path)
        else:
            model = torch.jit.trace(model, x, check_trace=False)
            model.save(trace_model_path)
            

    def torch2onnx(self,
                   model,
                   model_name,
                   data_shape,
                   extra_name=None,
                   save_dir=None,
                   verbose=False,
                   enable_fp16=False,
                   opset=13,
                   custom_opsets=None,
                   training=torch.onnx.TrainingMode.EVAL):
        # check disk
        usage_ratio = self.check_disk()
        # if usage_ratio >= 98:
            # raise OSError('Disk free space is not enough.')
        # elif usage_ratio >= 80:
            # warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        if save_dir is None:
            save_dir = self.save_root+"/onnx"
        
        os.makedirs(save_dir, exist_ok=True)
        if extra_name:
            model_name = model_name + '_' + extra_name
        onnx_path = os.path.join(save_dir, (model_name+'.onnx'))
        if isinstance(data_shape, dict):
            export_data = []
            for k, v in data_shape.items():
                export_data.append(torch.rand(v))
            if len(export_data) == 1:
                export_data = export_data[0]
        else:
            export_data = torch.rand(data_shape)
        torch.onnx.export(model,
                          export_data,
                          onnx_path,
                          opset_version=opset,
                          custom_opsets=custom_opsets,
                          input_names=['input'],
                          output_names=['output'],
                          verbose=False,
                          do_constant_folding=False,
                          training=training)
        if enable_fp16:
            print("convert 2 fp16")
            onnx_model = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(onnx_model)
            os.remove(onnx_path)
            onnx.save(model_fp16, onnx_path)
        timestamp('converter', 'convert to onnx over')


    def torch2ncnn(self,
                   model,
                   model_name,
                   data_shape,
                   save_dir=None,
                   verbose=False,
                   enable_int8=False,
                   check_disk=False,
                   enable_gemm_war=False):
        if check_disk:
            # check disk
            usage_ratio = self.check_disk()
            if usage_ratio >= 95:
                raise OSError('Disk free space is not enough.')
            elif usage_ratio >= 80:
                warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        if save_dir is None:
            save_dir = self.save_root+"/ncnn"
        
        model = model.eval()
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(data_shape, dict):
            x = []
            str_shape = ''
            for k, v in data_shape.items():
                x.append(torch.rand(v))
                if enable_gemm_war:
                    v = v[-1]
                if len(str_shape) == 0:
                    str_shape+=(str(v).replace(' ', ''))
                else:
                    str_shape+=(','+str(v).replace(' ', ''))
        else:
            x = torch.rand(data_shape)
            if enable_gemm_war:
                data_shape = data_shape[-1]
            if isinstance(data_shape, tuple):
                str_shape = str(data_shape).replace(' ', '')[1:-1]
            else:
                str_shape = str(data_shape).replace(' ', '')[1:-1]
        #print(x.shape)
        model = torch.jit.trace(model, x, check_trace=False)
        trace_model_path = os.path.join(save_dir, (model_name+'_tracing.pt'))

        model.save(trace_model_path)
        basic_savepath = os.path.join(save_dir, model_name)
        
        pipe = subprocess.DEVNULL
        if verbose:
            pipe = subprocess.STDOUT

        shutil.move(trace_model_path, os.path.basename(trace_model_path))
        result = subprocess.run(['pnnx',
                                  os.path.basename(trace_model_path),
                                  'inputshape='+str_shape],
                                 subprocess.PIPE,
                                 stderr=pipe)
        
        generated_files = ['.pnnx.onnx', '.ncnn.bin', '.ncnn.param', '_ncnn.py', '.pnnx.bin', '.pnnx.param', '_pnnx.py', '.pt']
        for f in generated_files:
            ori_path = os.path.basename(trace_model_path).split('.')[0]+f
            tar_path = os.path.join(save_dir, ori_path)
            shutil.move(ori_path, tar_path)

        if enable_int8:
            timestamp('gcWorker', "Using ncnn int8")
            table_path=save_dir+'/'+model_name+'.table'
            results = subprocess.run(['ncnn2table',
                                      save_dir+'/'+model_name+'_tracing.ncnn.param',
                                      save_dir+'/'+model_name+'_tracing.ncnn.bin',
                                      os.path.join(os.getenv('AUTOTAILOR_HOME'), 'infra/converter/imagelist.txt'),
                                      table_path,
                                      "mean=[104,117,123]",
                                      "norm=[0.017,0.017,0.017]",
                                      "shape=[32,32,64]",
                                      "pixel=BGR",
                                      "thread=1",
                                      "method=kl"],
                                     subprocess.PIPE,
                                     stderr=pipe)
            
            results = subprocess.run(['ncnn2int8',
                                      save_dir+'/'+model_name+'_tracing.ncnn.param',
                                      save_dir+'/'+model_name+'_tracing.ncnn.bin',
                                      save_dir+'/'+model_name+'_tracing_int8.ncnn.param',
                                      save_dir+'/'+model_name+'_tracing_int8.ncnn.bin',
                                      table_path
                                      ],
                                      subprocess.PIPE,
                                      stderr=pipe)
        else:
            logger.info("Disable int8")


        
    def onnx2torch(self, onnx_path, model_name=None, save_dir=None):
        # check disk
        # usage_ratio = self.check_disk()
        # if usage_ratio >= 98:
        #     raise OSError('Disk free space is not enough.')
        # elif usage_ratio >= 80:
        #     warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        if model_name is None:
            model_name = onnx_path.split('/')[-1].split('.')[0]
        if save_dir is None:
            save_dir = self.save_root+"/torch"

        torch_model = onnx2torch.convert(onnx_path)
        torch.save(torch_model, os.path.join(save_dir, model_name+'.pt'))
        return torch_model
        
    def onnx2ncnn(self, 
                  onnx_path, 
                  model_name=None, 
                  save_dir=None, 
                  passes=['eliminate_identity'], 
                  verbose=False):
        # check disk
        usage_ratio = self.check_disk()
        if usage_ratio >= 95:
            raise OSError('Disk free space is not enough.')
        elif usage_ratio >= 80:
            warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        if model_name is None:
            model_name = onnx_path.split('/')[-1].split('.')[0]
        if save_dir is None:
            save_dir = self.save_root+"/ncnn"

        sim_onnx_path = onnx_path.split('.')[0] + '_sim.onnx'
        
        onnx_model = onnx.load(onnx_path)
        optimized_model = oop.optimize(onnx_model, passes)
        onnx.save(optimized_model, sim_onnx_path)
        
        pipe = subprocess.DEVNULL
        if verbose:
            pipe = subprocess.STDOUT
        
        result = subprocess.run([os.path.join(os.getenv('AUTOTAILOR_HOME'), 'tools','onnx2ncnn'),
                                  sim_onnx_path,
                                  save_dir+'/'+model_name+'.param',
                                  save_dir+'/'+model_name+'.bin'],
                                 subprocess.PIPE,
                                 stderr=pipe)
    
    def tf2tflite(self,
                  tf_model,
                  model_name=None,
                  save_tf_dir=None,
                  save_tflite_dir=None,
                  enable_flex_ops=False,
                  enable_fp16=False,
                  enable_int8=False,
                  data_shape=None):
        # tf_model.save(save_tf_dir+'/'+model_name)
        # timestamp('converter', 'convert to tf over')
        tf_cv = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        tf_cv.optimizations = [tf.lite.Optimize.DEFAULT]
        if enable_fp16:
            logger.info("Use TFlite fp16")
            tf_cv.target_spec.supported_types = [tf.float16]

        elif enable_int8:
            logger.info("Use tflite int8")
            def _representative_data_gen():
                """Dataset generator that generates random tensor with the same shape as the input"""
                for _ in range(100):
                    yield [tf.random.uniform(shape=data_shape,dtype=tf.float32)]
            tf_cv.representative_dataset = _representative_data_gen
            tf_cv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            tf_cv.inference_input_type=tf.uint8
            tf_cv.inference_output_type=tf.uint8
            
        else:
            logger.info("Use tflite fp32")
        if enable_flex_ops:
            tf_cv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
            ]
        print(save_tflite_dir+'/'+model_name)
        open(save_tflite_dir+'/'+model_name+'.tflite','wb').write(
            tf_cv.convert()
        )
        timestamp('converter', 'convert to tflite over')
    
    def onnx2tflite(self,
                    onnx_path,
                    model_name=None,
                    save_tf_dir=None,
                    enable_flex_ops=False,
                    enable_fp32=True,
                    enable_fp16=False,
                    enable_int8=False,
                    data_shape=None,
                    convert_config=None,
                    verbose=True):
        only_fp32 = enable_fp32
        only_fp16 = enable_fp16 and not enable_int8
        only_int8 = enable_int8 and not enable_fp16
        assert only_fp32 or only_fp16 or only_int8
        # check disk
        # usage_ratio = self.check_disk()
        # if usage_ratio >= 98:
        #     raise OSError('Disk free space is not enough.')
        # elif usage_ratio >= 80:
        #     warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        if model_name is None:
            model_name = onnx_path.split('/')[-1].split('.')[0]
        if save_tf_dir is None:
            save_tf_dir = self.save_root+"/tf"
        
        tf_model = onnx2tf.convert(onnx_path,
                                   output_folder_path=save_tf_dir,
                                   output_integer_quantized_tflite=enable_int8,
                                   copy_onnx_input_output_names_to_tflite=True,
                                   verbosity='debug' if verbose else 'error',
                                   only_fp32=only_fp32,
                                   only_fp16=only_fp16,
                                   only_int8=only_int8,
                                   param_replacement_file=convert_config)
        timestamp('converter', 'convert to tf and tflite over')
        
        if enable_fp16:
            logger.info("Use TFlite fp16")

        elif enable_int8:
            logger.info("Use tflite int8")
            
        else:
            logger.info("Use tflite fp32")
        # if enable_flex_ops:
        #     tf_cv.target_spec.supported_ops = [
        #         tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        #         tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        #     ]
        # open(save_tflite_dir+'/'+model_name+'.tflite','wb').write(
        #     tf_cv.convert()
        # )

    def torch2tflite(self,
                     model,
                     model_name,
                     data_shape,
                     enable_fp32=True,
                     enable_fp16=False,
                     enable_int8=False,
                     save_dir=None,
                     verbose=False):
        self.torch2onnx(model,
                        model_name,
                        data_shape,
                        save_dir=save_dir,
                        training=torch.onnx.TrainingMode.PRESERVE,
                        verbose=verbose)
        onnx_path = os.path.join(save_dir, (model_name+'.onnx'))
        self.onnx2tflite(onnx_path,
                         save_tf_dir=save_dir,
                         enable_fp32=enable_fp32,
                         enable_fp16=enable_fp16,
                         enable_int8=enable_int8)
    
    def torch2script(self,
                     model,
                     model_name,
                     data_shape,
                     save_dir):
        if isinstance(data_shape, dict):
            x = []
            for k, v in data_shape.items():
                x.append(torch.rand(v))
        else:
            x = torch.rand(data_shape)
        trace_model_path = os.path.join(save_dir, (model_name+'_tracing.pt'))
        
        model = torch.jit.trace(model, x, check_trace=False)
        model.save(trace_model_path)
        
    def torch2coreml(self,
                     model,
                     model_name,
                     data_shape,
                     save_dir):
        if isinstance(data_shape, dict):
            x = []
            for k, v in data_shape.items():
                x.append(torch.rand(v))
        else:
            x = torch.rand(data_shape)
        
        model_path = os.path.join(save_dir, f"{model_name}.mlmodel")
        model = torch.jit.trace(model, x, check_trace=False)
        
        model = ct.convert(
            model,
            convert_to="neuralnetwork",
            inputs=[ct.TensorType(shape=x.shape, name="input")]
        )
        
        model.save(model_path)

    def check_disk(self):
        ps = subprocess.Popen(('df', '-h'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('grep', '/dev/nvme'), stdin=ps.stdout).decode('utf-8')
        ps.wait()
        
        return int(output.split(' ')[-2].split('%')[0])
