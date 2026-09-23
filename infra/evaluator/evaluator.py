import os
import json

from tqdm import tqdm

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# from deepspeed.profiling.flops_profiler import get_model_profile

from .metrics import accuracy, MetricLogger
import os
from infra.connector.connector import BackendConnector
from infra.convertor.convertor import Convertor
from autotailor.predictor_factory.profiler.profiler import Profiler


workspace = os.getenv("WORKSPACE")

class Evaluator:
    def __init__(self, config_file=None, config_dict=None):
        if config_file:
            with open(config_file, 'r') as f:
                self.config_dict = json.load(f)
            self.valdir = self.config_dict['valdir']
            self.batch_size = self.config_dict['batchsize']
            self.workers = self.config_dict['workers']
            self.device = self.config_dict['device']
            self.valdataset = None
        elif config_dict:
            self.config_dict = config_dict
            self.valdir = self.config_dict['valdir']
            self.valdataset = self.config_dict['valdataset']
            self.batch_size = self.config_dict['batchsize']
            self.workers = self.config_dict['workers']
            self.device = self.config_dict['device']
        self.transform_composes = None
        
    def evaluate(self):
        raise NotImplementedError

    def evaluate_latency(self,
                         model,
                         input_shape,
                         backend_config,
                         command_config,
                         verbose=False):
        backend_info = json.load(open(backend_config, 'r'))
        command_info = json.load(open(command_config, 'r'))
        backend = os.path.basename(command_config).split('_')[0]
        profiler = Profiler(backend_info, command_info)
        
        latency = profiler.profile_model(model.cpu(), input_shape)
        # # convert and send model
        # model_name = "test_model"
        # if backend == 'ncnn':
        #     convertor.torch2ncnn(model=model.cpu(),
        #                         model_name=model_name,
        #                         data_shape=input_shape,
        #                         save_dir='model_zoo/test_single_model',
        #                         verbose=True,
        #                         enable_int8=True if quant=='int8' else False)
        # elif backend == 'tflite':
        #     convertor.torch2tflite(model=model.cpu(),
        #                         model_name=model_name,
        #                         data_shape=input_shape,
        #                         verbose=False,
        #                         save_dir='model_zoo/test_single_model',
        #                         enable_fp32=True if quant=='fp32' else False,
        #                         enable_fp16=True if quant=='fp16' else False,
        #                         enable_int8=True if quant=='int8' else False)
        #     if quant=='fp32':
        #         model_name += '_float32'
        #     elif quant=='fp16':
        #         model_name += '_float16'
        #     else:
        #         raise NotImplementedError
        # elif backend == 'onnx':
        #     convertor.torch2onnx(model=model.cpu(),
        #                         model_name=model_name,
        #                         data_shape=input_shape,
        #                         verbose=False,
        #                         save_dir='model_zoo/test_single_model',
        #                         enable_fp16=True if quant=='fp16' else False)
        
        # converted_path = os.path.join("model_zoo/test_single_model", model_name)
    
        # # send
        # try:
        #     if backend == 'ncnn':
        #         base_dir = converted_path
        #         base_dir = base_dir.replace('-', '_')
        #         ncnn_param_path = base_dir + '_tracing.ncnn.param'
        #         ncnn_bin_path = base_dir + '_tracing.ncnn.bin'
        #         connector.send_model(ncnn_param_path)
        #         connector.send_model(ncnn_bin_path)
        #     elif backend == 'tflite':
        #         base_dir = converted_path
        #         tflite_path = base_dir + '.tflite'
        #         connector.send_model(tflite_path)
        #     elif backend == 'onnx':
        #         base_dir = converted_path
        #         onnx_path = base_dir + '.onnx'
        #         connector.send_model(onnx_path)
        # except Exception as e:
        #     error_save_path = os.path.join(workspace, 'results', 'generate_error.log')
        #     os.makedirs(os.path.join(workspace, 'results'), exist_ok=True)
        #     open(os.path.join(error_save_path), 'a').write(f'{model_name}: {e}\n')
        
        # # profile
        # if backend == 'tflite':
        #     model_name = model_name + '.tflite'
        # elif backend == 'ncnn':
        #     model_name = model_name + '_tracing.ncnn'
        # elif backend == 'onnx':
        #     model_name = model_name + '.onnx'
        # model_path = os.path.join(connector.model_dir, model_name)
        # configs = command_info
        
        # command_info["HW0"] = input_shape[2]
        # command_info["HW1"] = input_shape[3]
        # command_info["CIN"] = input_shape[1]
        
        # configs['shape'] = f'[{input_shape[2]},{input_shape[3]},{input_shape[1]}]'
        
        # if quant=='fp32':
        #     specific_benchmark = connector.benchmark_fp32_model_path 
        # elif quant=='fp16':
        #     specific_benchmark = connector.benchmark_fp16_model_path 
        # else:
        #     raise NotImplementedError
            
        
        # res = connector.profile(model_path=model_path,
        #                         configs=configs,
        #                         enable_latency_constraint=False,
        #                         verbose=True,
        #                         specific_benchmark=specific_benchmark)
        # latency = connector.parse(res)['latency']
        print(f"profile latency {latency}")
        
        return latency
    
    def evaluate_accuracy(self,
                          model,
                          type='ImageNet',
                          inp_resolution=224,
                          transform_composes=None):
        """
        Evaluating the accuracy of the given model based on the initialized
            dataset.

        Args:
            model (nn.Module): the model to be evaluated.
            type (str, optional): the type of preprocessing.
                Defaults to 'ImageNet'.

        Raises:
            NotImplementedError: If type not including, rasing the error.

        Returns:
            dict: all accuracy results in evaluating.
        """
        metric_logger = MetricLogger(delimiter="  ")
        
        if type == "ImageNet":
            if transform_composes:
                self.transform_composes = transform_composes
            else:
                self.transform_composes = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(inp_resolution),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])
        elif type == "CIFAR10":
            self.transform_composes = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise NotImplementedError
        
        if self.valdataset:
            self.val_dataset = self.valdataset
        else:
            self.val_dataset = datasets.ImageFolder(
                os.path.join(self.valdir, "val"),
                self.transform_composes
            )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )
        
        model.eval()
        model = model.to(self.device)
        with torch.no_grad():
            bs = None
            for images, labels in tqdm(self.val_loader):
                tmp_batch = images.shape[0]
                if bs is None:
                    bs = images.shape[0]
                if images.shape[0] != bs:
                    # Fix bs related tir updating problem
                    tiled_tensor = images.repeat(bs//images.shape[0]+1, 1, 1, 1)

                    # Slice the tensor to the desired shape [128, 3, 256, 256]
                    images = tiled_tensor[:bs]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = model(images)[:tmp_batch]
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                
                metric_logger.meters['acc1'].update(acc1.item(), n=bs)
                metric_logger.meters['acc5'].update(acc5.item(), n=bs)

        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
    def get_flops_and_params(self, model, test_input_size):
        """
        Profiling the model by DeepSpeed to get flops and parameter size of the
            model.

        Args:
            model (nn.Module): the model to be profiled.
            test_input_size (tuple): the shape of the input data in profiling.

        Returns:
            dict: flops and parameter size dictionary.
        """
        flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=test_input_size, # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=False, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
        print('FLOPs: ', str(flops), ' Parameter size: ', str(params))
        
        return {'flops': flops, 'params': params}