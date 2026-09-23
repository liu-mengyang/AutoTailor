# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import importlib


__BUILTIN_BACKENDS__ = {
    "ncnn_cpu": {
        "class_module": "infra.connector.backends.ncnn",
        "class_name": "NcnnCPUBackend",
        "default_benchmark_path": "benchncnn_fp32",
    },
    "ncnn_cpu_local": {
        "class_module": "infra.connector.backends.ncnn_local",
        "class_name": "NcnnLocalCPUBackend",
        "default_benchmark_path": "benchncnn_fp32",
    },
    "ncnn_gpu_local": {
        "class_module": "infra.connector.backends.ncnn_local",
        "class_name": "NcnnLocalGPUBackend",
        "default_benchmark_path": "benchncnn_fp32",
    },
    "ncnn_gpu": {
        "class_module": "infra.connector.backends.ncnn",
        "class_name": "NcnnGPUBackend",
        "default_benchmark_path": "benchncnn_fp32",
    },
    "tflite_cpu": {
        "class_module": "infra.connector.backends.tflite",
        "class_name": "TFLiteCPUBackend",
        "default_benchmark_path": "tflite",
    },
    "tflite_gpu": {
        "class_module": "infra.connector.backends.tflite",
        "class_name": "TFLiteGPUBackend",
        "default_benchmark_path": "tflite",
    },
}


class BaseBackend:
    """
    the base backend class to instantiate a backend instance. If users want to implement their own backend,
    the customized Backend should inherit this class.

    @params:

    profiler_class: a subclass inherit form `nn_meter.builder.backend.BaseProfiler` to specify the running command of
        the backend. A profiler contains commands to push the model to mobile device, run the model on the mobile device,
        get stdout from the mobile device, and related operations. In the implementation of a profiler, an interface of
        ``Profiler.profile()`` is required.
    
    parser_class: a subclass inherit form `nn_meter.builder.backend.BaseParser` to parse the profiled results.
        A parser parses the stdout from devices profiler and get required metrics. In the implementation of a parser, interface
        of `Parser.parse()` and property of `Parser.results()` are required.
    """
    profiler_class = None
    parser_class = None

    def __init__(self, configs):
        """ class initialization with required configs
        """
        self.configs = configs
        self.update_configs()
        if self.parser_class:
            self.parser = self.parser_class(**self.parser_kwargs)
        if self.profiler_class:
            self.profiler = self.profiler_class(**self.profiler_kwargs)

    def update_configs(self):
        """ update the config parameters for the backend
        """
        self.parser_kwargs = {}
        self.profiler_kwargs = {}
    
    def convert_model(self, model_path, save_path, input_shape = None):
        """ convert the Keras model instance to the type required by the backend inference.

        @params:
        
        model_path: the path of model waiting to profile
        
        save_path: folder to save the converted model
        
        input_shape: the shape of input tensor for inference, a random tensor according to the shape will be 
            generated and used
        """
        # convert model and save the converted model to path `converted_model`
        converted_model = model_path
        return converted_model

    def profile(self, converted_model, metrics = ['latency'], **kwargs):
        """
        run the model on the backend, return required metrics of the running results. nn-Meter only support latency
        for metric by now. Users may provide other metrics in their customized backend.

        @params:

        converted_model: the model path in type of backend required
        
        metrics: a list of required metrics name. Defaults to ['latency']
        
        """
        return self.parser.parse(self.profiler.profile(converted_model, **kwargs)).results.get(metrics)

    def profile_model_file(self, model_path, save_path, input_shape = None, metrics = ['latency'], **kwargs):
        """ load model by model file path, convert model file, and run ``self.profile()``
        @params:

        model_path: the path of model waiting to profile
        
        save_path: folder to save the converted model
        
        input_shape: the shape of input tensor for inference, a random tensor according to the shape will be 
            generated and used
        """
        converted_model = self.convert_model(model_path, save_path, input_shape)
        res = self.profile(converted_model, metrics, input_shape=input_shape, **kwargs)
        return res

    def test_connection(self):
        """ check the status of backend interface connection.
        """
        pass


class BaseProfiler:
    """
    Specify the profiling command of the backend. A profiler contains commands to push the model to mobile device, run the model 
    on the mobile device, get stdout from the mobile device, and related operations. 
    """
    def profile(self):
        """ Main steps of ``Profiler.profile()`` includes 1) push the model file to edge devices, 2) run models in required times
        and get back running results. Return the running results on edge device.
        """
        output = ''
        return output


class BaseParser:
    """
    Parse the profiled results. A parser parses the stdout from devices runner and get required metrics.
    """
    def parse(self, content):
        """ A string parser to parse profiled results value from the standard output of devices runner. This method should return the instance
        class itself.

        @params
        
        content: the standard output from device       
        """
        return self

    @property
    def results(self):
        """ warp the parsed results by ``ProfiledResults`` class from ``nn_meter.builder.backend_meta.utils`` and return the parsed results value.
        """
        pass


def list_backends():
    """ list all backends supported by nn-Meter, including builtin backends and registered backends
    """
    return list(__BUILTIN_BACKENDS__.keys())
