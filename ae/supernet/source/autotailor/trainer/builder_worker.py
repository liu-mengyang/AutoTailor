import math
from multiprocessing import Process
import os
# import pickle
import sys
import time

import dill as pickle

AUTOTAILOR_HOME = os.environ["AUTOTAILOR_HOME"]
sys.path.append(AUTOTAILOR_HOME)

from autotailor.tir.tailor_ir import TailorIR
from autotailor.tailor.tailor import Tailor


class SubnetBuilderWorker(Process):
    def __init__(self,
                 subnet_configs,
                 subnet_out_dict,
                 supernet_config_dict,
                 dataset="imagenet",
                 queue_budget=1000):
        super(SubnetBuilderWorker, self).__init__()
        self.subnet_configs = subnet_configs
        self.subnet_out_dict = subnet_out_dict
        self.queue_budget = queue_budget
        self.dataset = dataset
        
        tir = TailorIR(supernet_config_dict)
        tir.parse_graph()
        
        self.tailor = Tailor(tir)
        
    def run(self):
        # dataset-aware configuring
        if self.dataset == "imagenet":
            for epoch_id, sub_dict in self.subnet_configs.items():
                if epoch_id == "max" or epoch_id == "min":
                    continue
                for batch_id, subsub_dict in sub_dict.items():
                    dynamic_batch_num = len(subsub_dict)
                    for i in range(dynamic_batch_num):
                        subnet_code = subsub_dict[str(i)]
                        self.tailor.transform(subnet_code)
                        subnet_nn = self.tailor.tir.build()
                        
                        # waiting for consuming built subnets
                        # while self.subnet_out_queue.qsize() >= self.queue_budget:
                        #     time.sleep(1)
                        
                        self.subnet_out_dict[str(epoch_id)+str(batch_id)+str(i)] = pickle.dumps(subnet_nn)
                        print("Writed one subnet")
                
            