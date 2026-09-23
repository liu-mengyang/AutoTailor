import math
import random
import time

from loguru import logger
from tqdm import tqdm


def build_group_of_subnets(worker_id, config_dict_lst, tailor, dynamic_batch_size):
    """
    Deprecated.
    Build subnets in advance.
    """
    sub_subnet_dict_list = []
    cnt = 0
    start_time = time.time()
    for i, config_dict in enumerate(config_dict_lst):
        subnet_dict = {}
        for k in range(dynamic_batch_size):
            subnet_code = config_dict[str(k)]
            tailor.transform(subnet_code)
            subnet_nn = tailor.tir.build()
            subnet_dict[k] = subnet_nn
            cnt += 1
        sub_subnet_dict_list.append(subnet_dict)
        if i % (len(config_dict_lst) // 20) == 0 and cnt != 0:
            logger.info(f"{worker_id}: {i} / {len(config_dict_lst)} done in {time.time()-start_time} s ({(time.time()-start_time)/cnt} s/subnet)")

    return sub_subnet_dict_list


def sample_subnets(training_config,
                   tailor,
                   configs=None,
                   strategy=['sandwich']):
    """
    Sampling subnets before training for each batch of each epoch.
    """
    if configs is None:
        config_dict = {}

        num_epochs = training_config.epochs
        if training_config.dataset == "imagenet":
            total_sample = 1281167
        else:
            raise NotImplemented(f"Unsupport dataset: {training_config.dataset}")
        batch_size = training_config.batch_size
        dynamic_batch_size = training_config.dynamic_batch_size
        print(f"Dynamic batch size: {dynamic_batch_size}")
        steps_per_epoch = math.ceil(total_sample / batch_size)
        cnt = 0

        if "sandwich" in strategy:
            for k in ["max", "min"]:
                if k == "max":
                    subnet_code = tailor.supercode
                elif k == "min":
                    subnet_code = tailor.min_sample()
                config_dict[k] = subnet_code
        enable_onestep = "onestep" in strategy
        # The anchor of each onestep batch is a compound sample; its neighbours are
        # one-step (Hamming-1) transforms of that anchor.
        enable_compound = ("compound" in strategy) or enable_onestep
        for i in tqdm(range(num_epochs)):
            config_dict[str(i)] = {}
            for j in range(steps_per_epoch):
                config_dict[str(i)][str(j)] = {}
                temps = []
                for k in range(dynamic_batch_size):
                    subnet_seed = int("%d%.3d%.3d" % (i * steps_per_epoch + j, k, 0))
                    random.seed(subnet_seed)
                    if enable_onestep and k != 0:
                        # one-step neighbour of the batch anchor (resolution locked)
                        anchor = temps[0]
                        subnet_code = tailor.one_step_sample(anchor)
                        subnet_code["resolution"] = anchor["resolution"]
                        tries = 0
                        while subnet_code in temps and tries < 50:
                            subnet_code = tailor.one_step_sample(anchor)
                            subnet_code["resolution"] = anchor["resolution"]
                            tries += 1
                    else:
                        subnet_code = tailor.sample_subnet(compound=enable_compound)
                        if k != 0:
                            subnet_code["resolution"] = temps[0]["resolution"]
                        while subnet_code in temps:
                            # resample
                            subnet_code = tailor.sample_subnet(compound=enable_compound)
                            subnet_code["resolution"] = temps[0]["resolution"]
                    temps.append(subnet_code)
                    config_dict[str(i)][str(j)][str(k)] = subnet_code
                    cnt += 1
    else:
        config_dict = configs

    return config_dict
