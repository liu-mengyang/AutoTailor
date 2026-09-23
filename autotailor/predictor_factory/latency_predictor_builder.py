import argparse
import json
import os
import random
import sys
import time

from loguru import logger
import torch
import torch.utils.data as Data


AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')

sys.path.append(AUTOTAILOR_HOME)

from autotailor.tailor.eff_predictors.feature_parser import FeatureParser
from autotailor.tailor.eff_predictors.mlp_trainer import Trainer, split_train_test
from autotailor.tailor.eff_predictors.litepred.litepred_trainer import LitePredTrainer

class LatencyPredictorBuilder:
    def __init__(self,
                 profiler,
                 design_space_dict=None,
                 op_list=None,
                 op_type=None,
                 task_configs=None,
                 task_paths=None,
                 predictor_type='autotailor'
                 ):
        self.profiler = profiler
        self.feature_parser = FeatureParser()
        self.design_space_dict = design_space_dict
        self.task_configs = task_configs
        self.op_list = op_list
        self.op_type = op_type
        self.predictor_type = predictor_type
        self.task_dict = {}
        self.features = {}
        self.latencies = {}
        
        if design_space_dict:
            for op_type in op_list:
                print(op_type)
                tasks = design_space_dict[op_type]
                random.shuffle(tasks)
                # batchsize
                tasks_chunks = [tasks[x:x+5] for x in range(0, len(tasks), 5)]
                self.task_dict[op_type] = tasks_chunks
                self.features[op_type] = []
                self.latencies[op_type] = []
                

        if task_configs:
            self.op_list = []
            self.task_configs = {}
            self.task_paths = task_paths
            for op_type, op_configs in task_configs.items():
                self.op_list.append(op_type)
                self.task_configs[op_type] = op_configs
                tasks = op_configs
                random.shuffle(tasks)
                # batchsize
                tasks_chunks = tasks
                tasks_chunks = [tasks[x:x+5] for x in range(0, len(tasks), 5)]

                self.task_dict[op_type] = tasks_chunks
                self.features[op_type] = []
                self.latencies[op_type] = []

        
    def build_by_task_configs(self, 
                              mark, 
                              lut_predictor=None,
                              extract_ratio=1.0, 
                              exist_profiled_config=None, 
                              exist_time=None, 
                              time_budget=None, 
                              save_path=None, 
                              duration=1000, 
                              build=False, 
                              mode=None):
        lut_path_dict = {}
        if lut_predictor is None:
            start_time = time.time()
            profiled_res = {}
            if time_budget:
                # Profiling
                logger.info("Profiling under time budget ...")
                task_index = 0
                cur_profiled_time = duration
                training_cost = 0.0
                if exist_profiled_config:
                    for op_type, ops in exist_profiled_config.items():
                        for op in ops.values():
                            op_feature = self.feature_parser.parse(op_type, op["config"])
                            self.features[op_type].append(op_feature)
                            self.latencies[op_type].append(float(op['latency'].split(" +- ")[0]))
                while time.time() - start_time < (time_budget):
                    for op_type in self.task_dict:
                        sub_tasks = self.task_dict[op_type]
                        if task_index >= len(sub_tasks):
                            t = time.time()
                            logger.info(f"all data are consumed, cost {t-start_time} time")
                            kernel_path = f"{save_path}_{op_type}_{cur_profiled_time}_profiled_res.json"
                            with open(kernel_path, 'w+') as f:
                                json.dump(profiled_res, f, indent=4)
                            lut_path_dict[op_type] = kernel_path
                            logger.info(f"dataset length: {len(self.features[op_type])}")
                            trainx, testx, trainy, testy = split_train_test(self.features[op_type], self.latencies[op_type])
                            train_dataset = Data.TensorDataset(torch.Tensor(trainx), torch.Tensor(trainy))
                            eval_dataset = Data.TensorDataset(torch.Tensor(testx), torch.Tensor(testy))
                            if self.predictor_type == 'autotailor' or 'nnmeter':
                                trainer = Trainer(train_dataset=train_dataset,
                                                    eval_dataset=eval_dataset,
                                                    kernel_type=op_type,
                                                    epochs=350,
                                                    output_name=f"{mark}_{op_type}_{cur_profiled_time}")
                            elif self.predictor_type == 'litepred':
                                    trainer = LitePredTrainer(train_dataset=train_dataset,
                                                    eval_dataset=eval_dataset,
                                                    kernel_type=op_type,
                                                    epochs=350,
                                                    output_name=f"{mark}_{op_type}_{cur_profiled_time}")          
                
                            trainer.train()
                            trainer.save()
                            return
                        else:
                            model_configs = {}
                            for task in sub_tasks[task_index]:
                                id = f'{task[0]:06d}'
                                print(task)
                                path = self.task_paths[op_type][id]['converted_model']
                                if exist_profiled_config and id in exist_profiled_config[op_type].keys() and 'latency' in exist_profiled_config[op_type][id].keys():
                                    profiled_res[id] = {'kernel':op_type, 'config':task[1], 'converted_model': path, 'latency': exist_profiled_config[op_type][id]['latency']}
                                    logger.info(f"skip kernel {id}")
                                else:
                                    model_configs[id] = {'kernel':op_type, 'config':task[1], 'converted_model': path}
                            profiled_model_configs = self.profiler.profile_batch_models(model_configs,
                                                                                   op_type)

                            print(profiled_model_configs)
                            for key, value in profiled_model_configs.items():
                                op_feature = self.feature_parser.parse(op_type, value["config"])
                                self.features[op_type].append(op_feature)
                            
                                latency_str = str(value["latency"])
                                value["latency"] = latency_str
                                if "+-" in latency_str:
                                    self.latencies[op_type].append(float(latency_str.split(" +- ")[0]))
                                else:
                                    self.latencies[op_type].append(None)
                                    logger.info("generate kernel is segmentation fault")
                                
                                profiled_res[key] = value

                            remain_time = cur_profiled_time  - time.time() + start_time 
                            print(f"this duration remain {remain_time}")
                            task_index += 1
                            if time.time() - start_time - training_cost >= cur_profiled_time:
                                training_time = time.time()
                                profiled_time = cur_profiled_time + exist_time
                                kernel_save_path = f"{save_path}_{op_type}_{profiled_time}_profiled_res.json"
                                with open(kernel_save_path, 'w+') as f:
                                    json.dump(profiled_res, f, indent=4)
                                lut_path_dict[op_type] = kernel_save_path
                                logger.info(f"dataset length: {len(self.features[op_type])}")
                                trainx, testx, trainy, testy = split_train_test(self.features[op_type], self.latencies[op_type])
                                train_dataset = Data.TensorDataset(torch.Tensor(trainx), torch.Tensor(trainy))
                                eval_dataset = Data.TensorDataset(torch.Tensor(testx), torch.Tensor(testy))
                                if self.predictor_type == 'autotailor' or 'nnmeter':
                                    trainer = Trainer(train_dataset=train_dataset,
                                                    eval_dataset=eval_dataset,
                                                    kernel_type=op_type,
                                                    epochs=350,
                                                    output_name=f"{mark}_{op_type}_{profiled_time}")
                                elif self.predictor_type == 'litepred':
                                    trainer = LitePredTrainer(train_dataset=train_dataset,
                                                    eval_dataset=eval_dataset,
                                                    kernel_type=op_type,
                                                    epochs=350,
                                                    output_name=f"{mark}_{op_type}_{profiled_time}")          
                                cur_profiled_time += duration 
                                trainer.train()
                                trainer.save()
                                
                                training_cost += (time.time() - training_time)
                                logger.info(f"Training Cost Total: {training_cost}")
                            if time.time() - start_time - training_cost >= (time_budget - exist_time):
                                return
                            
            else:
                # Profiling
                logger.info("Preparing LUTs ...")
                
                for op_type in self.task_configs.keys():
                    logger.info(f"Preparing LUTs for {op_type} ...")
                    profiled_res = {}
                    s = time.time()
                    print(op_type)
                    model_configs = {}
                    for task in self.task_configs[op_type]:
                        id = f'{task[0]:06d}'
                        path = self.task_paths[op_type][id]['converted_model']
                        if exist_profiled_config and id in exist_profiled_config[op_type].keys() and 'latency' in exist_profiled_config[op_type][id].keys():
                            profiled_res[id] = {'kernel':op_type, 'config':task[1], 'converted_model': path, 'latency': exist_profiled_config[op_type][id]['latency']}
                            logger.info(f"skip kernel {id}")
                        else:
                            model_configs[id] = {'kernel':op_type, 'config':task[1], 'converted_model': path}
                    
                    profiled_model_configs = self.profiler.profile_batch_models(model_configs,
                                                                                   op_type, mode=mode)

                    for key, value in profiled_model_configs.items():
                        if op_type != 'Conv' and op_type != 'DepthConv':
                            self.features[op_type].append(value['config'])
                        else:
                            op_feature = self.feature_parser.parse(op_type, value["config"])
                            self.features[op_type].append(op_feature)
                            
                        latency_str = str(value["latency"])
                        value["latency"] = latency_str
                        if "+-" in latency_str:
                            self.latencies[op_type].append(float(latency_str.split(" +- ")[0]))
                        else:
                            self.latencies[op_type].append(None)
                        
                        profiled_res[key] = value
                    
                    kernel_save_path = f"{save_path}_{op_type}_profiled_res.json"
                    with open(kernel_save_path, 'w+') as f:
                        json.dump(profiled_res, f, indent=4)
                    lut_path_dict[op_type] = kernel_save_path
                    e = time.time()
                    t = e - s
                    logger.info(f"Preparing {op_type} LUTs cost {t} s")
                    lut_dict_path = f"{save_path}_lut_dict.json"
                    logger.info(f"Saving LUT path dict ... in {lut_dict_path}")
                    if os.path.exists(lut_dict_path):
                        with open(lut_dict_path, 'r') as f:
                            existing_lut_dict = json.load(f)
                        existing_lut_dict.update(lut_path_dict)
                        lut_path_dict = existing_lut_dict
                    logger.info(f"LUT path dict: {lut_path_dict}")
                    with open(lut_dict_path, 'w') as f:
                        json.dump(lut_path_dict, f, indent=4)
                if build == False:
                    return
            # json.dump(open("{save_path}_lut_dict.json", 'w'))
            # with open(f"{save_path}_lut_dict.json", 'w') as f:
            #     json.dump(lut_path_dict, f, indent=4)
            lut_dict_path = f"{save_path}_lut_dict.json"
            logger.info(f"Saving LUT path dict ... in {lut_dict_path}")
            if os.path.exists(lut_dict_path):
                with open(lut_dict_path, 'r') as f:
                    existing_lut_dict = json.load(f)
                existing_lut_dict.update(lut_path_dict)
                lut_path_dict = existing_lut_dict
            logger.info(f"LUT path dict: {lut_path_dict}")
            with open(lut_dict_path, 'w') as f:
                json.dump(lut_path_dict, f, indent=4)
        else:
            logger.info(f"Extracting for building [ratio: {extract_ratio}] ...")
            # extracting
            for op_type in self.task_dict:
                sub_tasks = self.task_dict[op_type]
                tasks = sub_tasks[:int(len(sub_tasks)*extract_ratio)]
                for task in tasks:
                    for feature in task:
                        feature = tuple(tuple(sub) if isinstance(sub, list) else sub for sub in feature)
                        print(feature)
                        latency = lut_predictor.lut_dict[op_type][feature] 
                        self.features[op_type].append(self.feature_parser.parse(op_type,
                                                                    feature))
                        self.latencies[op_type].append(latency)
                
        # Building
        for op_type in self.features:
            trainx, testx, trainy, testy = split_train_test(self.features[op_type], self.latencies[op_type])
            train_dataset = Data.TensorDataset(torch.Tensor(trainx), torch.Tensor(trainy))
            eval_dataset = Data.TensorDataset(torch.Tensor(testx), torch.Tensor(testy))
            if self.predictor_type == 'autotailor' or 'nnmeter':
                trainer = Trainer(train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                kernel_type=op_type,
                                epochs=350,
                                output_name=f"{mark}_{op_type}")
            elif self.predictor_type == 'litepred':
                trainer = LitePredTrainer(train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                kernel_type=op_type,
                                epochs=350,
                                output_name=f"{mark}_{op_type}")          
                
            trainer.train()
            trainer.save()
        
    def build(self, mark, lut_predictor=None, extract_ratio=1.0, time_budget=None, save_path=None, duration=1000):
        if lut_predictor is None:
            start_time = time.time()
            if time_budget:
                # Profiling
                logger.info("Profiling under time budget ...")
                task_index = 0
                cur_profiled_time = duration
                profiled_res = {}
                kernel_index = 0
                while time.time() - start_time < time_budget:
                    for op_type in self.task_dict:
                        sub_tasks = self.task_dict[op_type]
                        if task_index >= len(sub_tasks):
                            t = time.time()
                            logger.info(f"all data are consumed, cost {t-start_time} time")
                            break
                        profiled_model_configs = self.profiler.profile(op_type,
                                                                mark,
                                                                sub_tasks[task_index],
                                                                save_results=False)

                        for value in list(profiled_model_configs.values()):
                            op_feature = self.feature_parser.parse(op_type, value["config"])
                            self.features[op_type].append(op_feature)
                            
                            latency_str = str(value["latency"])
                            if "+-" in latency_str:
                                self.latencies[op_type].append(float(latency_str.split(" +- ")[0]))
                            else:
                                self.latencies[op_type].append(None)
                            
                            id = f'{kernel_index:06d}'
                            kernel_index += 1
                            value["latency"] = latency_str
                            profiled_res[id] = value

                        remain_time = cur_profiled_time - time.time() + start_time 
                        print(f"this duration remain {remain_time}")
                        task_index += 1
                        if time.time() - start_time >= cur_profiled_time:
                            with open(f"{save_path}_{op_type}_{cur_profiled_time}_profiled_res.json", 'w+') as f:
                                json.dump(profiled_res, f, indent=4)
                            cur_profiled_time += duration
                            trainx, testx, trainy, testy = split_train_test(self.features[op_type], self.latencies[op_type])
                            train_dataset = Data.TensorDataset(torch.Tensor(trainx), torch.Tensor(trainy))
                            eval_dataset = Data.TensorDataset(torch.Tensor(testx), torch.Tensor(testy))
                            if self.predictor_type == 'autotailor' or 'nnmeter':
                                trainer = Trainer(train_dataset=train_dataset,
                                                eval_dataset=eval_dataset,
                                                kernel_type=op_type,
                                                epochs=350,
                                                output_name=f"{mark}_{op_type}_{cur_profiled_time}")
                            elif self.predictor_type == 'litepred':
                                trainer = LitePredTrainer(train_dataset=train_dataset,
                                                eval_dataset=eval_dataset,
                                                kernel_type=op_type,
                                                epochs=350,
                                                output_name=f"{mark}_{op_type}_{cur_profiled_time}")          
                
                            trainer.train()
                            trainer.save()
                            
            else:
                # Profiling
                logger.info("Preparing LUTs ...")
                
                if self.design_space_dict:
                    op_type_list = list(self.design_space_dict.keys())
                elif self.task_configs:
                    op_type_list = list(self.task_configs.keys())
                
                for op_type in op_type_list:
                    kernel_index = 0
                    s = time.time()
                    profiled_model_configs = self.profiler.profile(op_type,
                                    mark,
                                    self.design_space_dict[op_type])
                    
                    for value in list(profiled_model_configs.values()):
                        latency_str = str(value["latency"])
                        id = f'{kernel_index:06d}'
                        kernel_index += 1
                        value["latency"] = latency_str
                        profiled_res[id] = value
                    
                    with open(f"{save_path}_{op_type}_profiled_res.json", 'w+') as f:
                            json.dump(profiled_res, f, indent=4)
                    e = time.time()
                    t = e - s
                    logger.info(f"Preparing {op_type} LUTs cost {t} s")
        else:
            logger.info(f"Extracting for building [ratio: {extract_ratio}] ...")
            # extracting
            for op_type in self.task_dict:
                sub_tasks = self.task_dict[op_type]
                tasks = sub_tasks[:int(len(sub_tasks)*extract_ratio)]
                for task in tasks:
                    for feature in task:
                        feature = tuple(tuple(sub) if isinstance(sub, list) else sub for sub in feature)
                        latency = lut_predictor.lut_dict[op_type][feature]
                        if latency == None:
                            print(feature)
                            print("zero latency")
                        else:
                            print(latency)
                        self.features[op_type].append(self.feature_parser.parse(op_type,
                                                                    feature))
                        self.latencies[op_type].append(latency)
                
        # Building
        for op_type in self.features:
            trainx, testx, trainy, testy = split_train_test(self.features[op_type], self.latencies[op_type])
            train_dataset = Data.TensorDataset(torch.Tensor(trainx), torch.Tensor(trainy))
            eval_dataset = Data.TensorDataset(torch.Tensor(testx), torch.Tensor(testy))
            if self.predictor_type == 'autotailor' or 'nnmeter':
                trainer = Trainer(train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                kernel_type=op_type,
                                epochs=350,
                                output_name=f"{mark}_{op_type}")
            elif self.predictor_type == 'litepred':
                trainer = LitePredTrainer(train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                kernel_type=op_type,
                                epochs=350,
                                output_name=f"{mark}_{op_type}")          
                
            trainer.train()
            trainer.save()
        
    def evaluate(self, dataset_dict, weight, op_type):
        features = []
        latencies = []
        if op_type in dataset_dict.keys():
            dataset_dict = dataset_dict[op_type]
        for value in dataset_dict.values():
            # print(value)
            feature = value['config']
            feature = tuple(tuple(sub) if isinstance(sub, list) else sub for sub in feature)
            if isinstance(value['latency'], str):
                latency = float(value['latency'].split(' +- ')[0])
            else:
                latency = value['latency']
                        
            features.append(self.feature_parser.parse(op_type,feature))
            latencies.append(latency)
        
       
        trainx, testx, trainy, testy = split_train_test(features, latencies)
        train_dataset = Data.TensorDataset(torch.Tensor(trainx), torch.Tensor(trainy))
        eval_dataset = Data.TensorDataset(torch.Tensor(testx), torch.Tensor(testy))
        if self.predictor_type == 'autotailor' or 'nnmeter':
            trainer = Trainer(train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                kernel_type=op_type,
                                weights=torch.load(weight, map_location='cpu'),
                                epochs=350
                                )
        elif self.predictor_type == 'litepred':
            trainer = LitePredTrainer(train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                kernel_type=op_type,
                                weights=torch.load(weight, map_location='cpu'),
                                epochs=350)  
        rmse_total, rmspe_total, error_total, acc5_total, acc10_total, acc15_total = trainer.evaluate(sample_num=len(dataset_dict.keys()))
        return rmse_total, rmspe_total, error_total, acc5_total, acc10_total, acc15_total