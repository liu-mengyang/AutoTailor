import copy
import math
import multiprocessing
import itertools as it
import json

import time
import toml
from tqdm import tqdm
from pprint import pprint
from loguru import logger
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def reduced_block_type(tir, stage_id, block_types):
    # find the block_type can be reduced in one stage
    stage = tir.stages[stage_id]
    for i in range(len(stage.flow)):
        block_type = stage.flow[len(stage.flow)-i-1].type
        if block_type in block_types:
            return block_type

    return None


def generate_candidates(tailor, pruning_kernel_size=False, maximal_depth=False):
    # num_workers = 1
    # num_workers = 32
    num_workers = multiprocessing.cpu_count()
    # prepare glob candidates
    glob_candidates = []

    glob_k_list = list(tailor.global_vars.keys())
    glob_v_list = []
    for glob_k in glob_k_list:
        glob_v = list(tailor.global_vars[glob_k])
        glob_v_list.append(glob_v)
    glob_v_comb_list = list(it.product(*glob_v_list))

    for glob_v_comb in glob_v_comb_list:
        code = {}
        for v_id, glob_v in enumerate(glob_v_comb):
            code[glob_k_list[v_id]] = glob_v
        glob_candidates.append(code)

    supercode = tailor.supercode

    # prepare stage candidate
    stage_candidates = []
    num_stage = tailor.num_stage

    # Skip some stage
    skipcode = {}
    for stage_k, stage_v in tailor.stage_vars.items():
        if "skipped" in stage_k:
            dim_name = stage_k.split("_skipped")[0]
            skipcode[dim_name] = stage_v

    # transform skipcode to positive values
    for dim_name, skipids in skipcode.items():
        for i in range(len(skipids)):
            if skipcode[dim_name][i] < 0:
                skipcode[dim_name][i] += num_stage
    print(f"Skipcode: {skipcode}")
    stage_k_list = list(tailor.stage_vars.keys())

    # filter out skipped
    temp_list = []
    for k in stage_k_list:
        if "skipped" in k:
            continue
        temp_list.append(k)
    stage_k_list = temp_list


    stage_v_list = []
    for stage_k in stage_k_list:
        if stage_k == "reduce_depth":
            # skip reduce depth for only using minimal depth
            continue
        stage_v = list(tailor.stage_vars[stage_k])
        stage_v_list.append(stage_v)
    initial_skip_stages = None
    for dim_name, skip_stages in skipcode.items():
        if dim_name == "reduce_depth":
            continue
        if initial_skip_stages is None:
            # init
            initial_skip_stages = set(skip_stages)
        else:
            initial_skip_stages = set(skip_stages) & initial_skip_stages

    num_skip_stages = num_stage
    if initial_skip_stages:
        initial_skip_stages = list(initial_skip_stages)
        num_skip_stages = len(initial_skip_stages)

        stage_v_comb_list = list(it.product(*stage_v_list, repeat=num_stage - num_skip_stages))
    else:
        # only reduce depth in stage vars
        stage_v_comb_list = [supercode["reduce_depth"]]
    if "reduce_depth" in tailor.stage_vars:
        reduce_num = -min(tailor.stage_vars["reduce_depth"])
    reduced_block_types = []
    num_k = len(stage_k_list)
    minimal_reduce_depth = []
    for j in range(tailor.num_stage):
        reduced_block_types.append(reduced_block_type(tailor.tir, j, list(tailor.block_vars.keys())))
    print(f"Reduced blocks: {reduced_block_types}")
    if "reduce_depth" in tailor.stage_vars:
        # at least keep 1 block of this reduce block type and 2 blocks in this stage
        index_depth = stage_k_list.index("reduce_depth")


        for i in range(num_stage):
            if i not in skipcode["reduce_depth"]:
                stage_num_blocks_total = 0
                # count # of blocks in this stage
                block_types = list(tailor.block_vars.keys())
                for block_type in block_types:
                    stage_num_blocks_total += len(list(supercode[block_type].values())[0][i])
                # print(i)
                stage_num_blocks = len(list(supercode[reduced_block_types[i]].values())[0][i])
                # print(f"{i}: {stage_num_blocks}")
                reduce_num_temp = min(min(reduce_num, stage_num_blocks-1), stage_num_blocks_total-2)
                minimal_reduce_depth.append(-reduce_num_temp)
            else:
                minimal_reduce_depth.append(0)
    print(f"Minimal reduce depth: {minimal_reduce_depth}")

    print(f"# of stage: {num_stage}")
    print(f"len of stage v comb before padding: {len(stage_v_comb_list)}")

    if "reduce_depth" in stage_k_list:
        num_k -= 1
    for stage_v_comb in stage_v_comb_list:
        stage_v_comb = list(stage_v_comb)

        code = {}
        temp_code = {}
        for i, stage_k in enumerate(stage_k_list):
            # select the code of this key from comb list
            if stage_k != "reduce_depth":
                temp_code[stage_k] = stage_v_comb[i::num_k]
            # skip some stage
            code[stage_k] = copy.deepcopy(supercode[stage_k])
            index = 0
            for j in range(len(code[stage_k])):
                if initial_skip_stages:
                    if j not in initial_skip_stages and j not in skipcode[stage_k]:
                        if stage_k == "reduce_depth":
                            if maximal_depth:
                                code["reduce_depth"][j] = 0
                            else:
                                code[stage_k][j] = max(min(tailor.stage_vars["reduce_depth"]), minimal_reduce_depth[j])
                        else:
                            code[stage_k][j] = temp_code[stage_k][index]
                        index += 1
                    # else they keep supercode
                else:
                    assert stage_k == "reduce_depth"
                    if j not in skipcode["reduce_depth"]:
                        if maximal_depth:
                            code["reduce_depth"][j] = 0
                        else:
                            code["reduce_depth"][j] = max(min(tailor.stage_vars["reduce_depth"]), minimal_reduce_depth[j])
        if code not in stage_candidates:
            stage_candidates.append(code)
            # print(code)
        # print(stage_candidates)
    print(f"len of stage v comb after padding: {len(stage_candidates)}")

    # prepare block candidate
    block_candidates = []
    block_type_list = list(tailor.block_vars.keys())

    block_v_comb_list_dict = {}
    for block_type in block_type_list:
        print(f"Start to handle {block_type}")
        sub_block_candidates = []
        block_k_list = list(tailor.block_vars[block_type].keys())

        block_v_list = []
        for i, block_k in enumerate(block_k_list):
            if block_k == "kernel_size" and pruning_kernel_size:
                # reduce kernel size
                continue
            block_v = list(tailor.block_vars[block_type][block_k])
            block_v_list.append(block_v)
            print(f"{block_k}: {block_v}")

        # count num of blocks
        num_blocks = 0
        supercode_list = tailor.supercode[block_type][block_k] # k-agnostic

        block_v_comb_list = []
        num_candidates = 0
        for i in range(len(supercode_list)):
            stage_num_blocks_total = 0
            # count # of blocks in this stage
            block_types = list(tailor.block_vars.keys())
            for block_type_temp in block_types:
                stage_num_blocks_total += len(list(supercode[block_type_temp].values())[0][i])

            stage_num_blocks = 0
            for j in range(len(supercode_list[i])):
                stage_num_blocks += 1
            if not maximal_depth:
                if "reduce_depth" in tailor.stage_vars:
                    # keep minimal depth
                    if i not in skipcode["reduce_depth"] and block_type == reduced_block_types[i]:
                        # at least keep 1 block but 2 blocks in this stage
                        reduce_num_temp = min(min(reduce_num, stage_num_blocks-1), max(stage_num_blocks_total-2, 0))
                        stage_num_blocks -= reduce_num_temp
                        print(f"Stage {i} has {stage_num_blocks} blocks in extraction")
            num_blocks += stage_num_blocks

            stage_block_v_comb_list = list(it.product(*block_v_list, repeat=stage_num_blocks))
            block_v_comb_list.append(stage_block_v_comb_list)
            num_candidates += len(stage_block_v_comb_list)

        # imrpoved by stage-level independent
        # block_v_comb_list = list(it.product(*block_v_list, repeat=num_blocks))
        # print(block_v_comb_list)
        print(f"# of block v comb: {num_candidates}")
        print(f"# of block: {num_blocks}")

        num_k = len(block_k_list)
        # worker_tasks = []

        # # map
        # for i in range(num_workers):
        #     worker_tasks.append((i,
        #                          block_v_comb_list[i::num_workers],
        #                          block_k_list,
        #                          stage_k_list,
        #                          block_type,
        #                          tailor.supercode,
        #                          skipcode,
        #                          tailor.stage_vars,
        #                          tailor.block_vars,
        #                          tailor.num_stage,
        #                          reduced_block_types,
        #                          reduce_num,
        #                          pruning_kernel_size))

        # processing_st = time.time()

        # pool = multiprocessing.Pool(processes=num_workers)
        # sub_block_candidates_list = pool.starmap(block_generate_worker,
        #                                          worker_tasks)
        # pool.close()
        # pool.join()

        for stage_id, stage_block_v_comb_list in enumerate(block_v_comb_list):
            for block_v_comb in stage_block_v_comb_list:
                block_v_comb = list(block_v_comb)
                # print(block_v_comb)

                code = {}
                meet_kernel_size = False
                offset = 0

                for i in range(num_k):
                    if block_k_list[i] == "kernel_size" and pruning_kernel_size:
                        # skip kernel size
                        meet_kernel_size = True
                        code["kernel_size"] = copy.deepcopy(supercode[block_type]["kernel_size"])
                    else:
                        if meet_kernel_size:
                            offset = 1 # block_v_comb has no kernel_size comb
                        if "kernel_size" in tailor.block_vars[block_type] and pruning_kernel_size:
                            bias = 1
                        else:
                            bias = 0
                        # print(f"i: {i}")
                        # print(f"offset: {offset}")
                        # print(f"num_k: {num_k}")
                        # print(f"bias: {bias}")
                        block_v = block_v_comb[i-offset::(num_k-bias)]
                        # print(block_v)
                        block_k = block_k_list[i]
                        code[block_k] = copy.deepcopy(supercode[block_type][block_k])
                        v_id = 0
                        # for j in range(num_stage):
                        stage_num_blocks_total = 0
                        # count # of blocks in this stage
                        block_types = list(tailor.block_vars.keys())
                        for block_type_temp in block_types:
                            stage_num_blocks_total += len(list(supercode[block_type_temp].values())[0][stage_id])
                        # print(f"{j}: {stage_num_blocks_total}")
                        stage_num_block = len(code[block_k][stage_id])
                        if not maximal_depth and "reduce_depth" in stage_k_list and block_type == reduced_block_types[stage_id] and reduce_num and stage_id not in skipcode["reduce_depth"]:
                            # keep minimal depth
                            assert reduce_num >= 0
                            reduce_num_temp = min(min(reduce_num, stage_num_block-1), max(stage_num_blocks_total-2, 0))
                            # print(f"{j}: {stage_num_block} {reduce_num_temp}")
                            for k in range(stage_num_block-reduce_num_temp):
                                code[block_k][stage_id][k] = block_v[v_id]
                                v_id += 1
                        else:
                            for k in range(stage_num_block):
                                code[block_k][stage_id][k] = block_v[v_id]
                                v_id += 1

                final_code = {}
                final_code[block_type] = copy.deepcopy(code)
                # print(final_code)
                if final_code not in sub_block_candidates:
                    sub_block_candidates.append(final_code)

            # processing_time = time.time() - processing_st
            # # reduce
            # reduce_st = time.time()
            # sub_block_candidates = []

            # while len(sub_block_candidates_list) > 1:
            #     # map
            #     cnt = 0
            #     num_workers = math.ceil(len(sub_block_candidates_list)/2)
            #     worker_tasks = []
            #     for i in range(num_workers):
            #         if cnt + 1 < num_workers:
            #             worker_tasks.append((i,
            #                                 sub_block_candidates_list[cnt],
            #                                 sub_block_candidates_list[cnt+1]))
            #             cnt += 2
            #         else:
            #             # left one candidate
            #             worker_tasks.append((i,
            #                                 sub_block_candidates_list[cnt],
            #                                 None))
            #             cnt += 1

            #     processing_st = time.time()

            #     pool = multiprocessing.Pool(processes=num_workers)
            #     sub_block_candidates_list = pool.starmap(candidates_reduce_worker,
            #                                             worker_tasks)
            #     pool.close()
            #     pool.join()
            # sub_block_candidates = sub_block_candidates_list[0]
            # for sub_block_candidates_temp in sub_block_candidates_list:
            #     for sub_block_candidate in sub_block_candidates_temp:
            #         if sub_block_candidate not in sub_block_candidates:
            #             sub_block_candidates.append(sub_block_candidate)

            block_candidates.append(sub_block_candidates)
            # reduce_time = time.time() - reduce_st

    print("-----SUMMARY-----")
    # print(f"{processing_time} for distributed processing with {num_workers} workers.")
    # print(f"{reduce_time} for reduce.")
    print(f"# of global vars: {len(glob_candidates)}")
    print(f"# of stage vars: {len(stage_candidates)}")
    for candidates in block_candidates:
        print(f"# of candidate {list(candidates[0].keys())[0]} block vars: {len(candidates)}")

    num_block_type = len(block_candidates)
    # block_comb_candidates = list(it.product(*block_candidates))
    # Reduce cost: all block type are independent
    block_comb_candidates = []
    for candidates in block_candidates:
        block_type = list(candidates[0].keys())[0]
        for candidate in candidates:
            other_block = []
            for k, v in tailor.supercode.items():
                if isinstance(v, dict):
                    # block type var
                    if k == block_type:
                        continue
                    other_block.append({k:v})
            block_comb_candidates.append((candidate, *other_block))
    print(f"# of block vars after pruning: {len(block_comb_candidates)}")
    # print(block_comb_candidates)
    return glob_candidates, stage_candidates, block_comb_candidates


def candidates_reduce_worker(worker_id,
                             candidates_a,
                             candidates_b,
                             ):
    print(f"Reduce worker {worker_id} created")
    if candidates_b is None:
        print(f"Worker {worker_id} done")
        return candidates_a

    ret_candidates = []
    for candidate_a in candidates_a:
        if candidate_a not in candidates_b:
            ret_candidates.append(candidate_a)
    ret_candidates += candidates_b
    print(f"Reduce worker {worker_id} done")
    return ret_candidates


def block_generate_worker(worker_id,
                          block_v_comb_list,
                          block_k_list,
                          stage_k_list,
                          block_type,
                          supercode,
                          skipcode,
                          stage_vars,
                          block_vars,
                          num_stage,
                          reduced_block_types,
                          reduce_num,
                          pruning_kernel_size=False):
    num_k = len(block_k_list)

    sub_block_candidates = []
    num_tasks = len(block_v_comb_list)
    print(f"Worker {worker_id} created with {num_tasks} tasks")
    for block_v_comb in block_v_comb_list:
        block_v_comb = list(block_v_comb)

        code = {}
        meet_kernel_size = False
        offset = 0
        for i in range(num_k):
            if block_k_list[i] == "kernel_size" and pruning_kernel_size:
                # skip kernel size
                meet_kernel_size = True
                code["kernel_size"] = copy.deepcopy(supercode[block_type]["kernel_size"])
            else:
                if meet_kernel_size:
                    offset = 1 # block_v_comb has no kernel_size comb
                if "kernel_size" in block_vars[block_type] and pruning_kernel_size:
                    bias = 1
                else:
                    bias = 0

                block_v = block_v_comb[i-offset::(num_k-bias)]
                block_k = block_k_list[i]
                code[block_k] = copy.deepcopy(supercode[block_type][block_k])
                v_id = 0
                for j in range(num_stage):
                    stage_num_blocks_total = 0
                    # count # of blocks in this stage
                    block_types = list(block_vars.keys())
                    for block_type_temp in block_types:
                        stage_num_blocks_total += len(list(supercode[block_type_temp].values())[0][j])
                    # print(f"{j}: {stage_num_blocks_total}")
                    stage_num_block = len(code[block_k][j])
                    if "reduce_depth" in stage_k_list and block_type == reduced_block_types[j] and reduce_num and j not in skipcode["reduce_depth"]:
                        # keep minimal depth
                        assert reduce_num >= 0
                        reduce_num_temp = min(min(reduce_num, stage_num_block-1), max(stage_num_blocks_total-2, 0))
                        # print(f"{j}: {stage_num_block} {reduce_num_temp}")
                        for k in range(stage_num_block-reduce_num_temp):
                            code[block_k][j][k] = block_v[v_id]
                            v_id += 1
                    else:
                        for k in range(stage_num_block):
                            code[block_k][j][k] = block_v[v_id]
                            v_id += 1
        final_code = {}
        final_code[block_type] = copy.deepcopy(code)
        if final_code not in sub_block_candidates:
            sub_block_candidates.append(final_code)
    print(f"Worker {worker_id} done")
    return sub_block_candidates


def extractor_worker(worker_id, extractor, worker_task, task_code, block_level=False):
    extractor.clear_op_dict()
    num_tasks = len(worker_task)
    print(f"Worker {worker_id} created with {num_tasks} tasks")
    task_codes = []
    for i, stage_block_candidate in enumerate(worker_task):
        code = copy.deepcopy(task_code)
        for var_candidate in stage_block_candidate:
            if isinstance(var_candidate, tuple):
                # block type
                for block_type_candidate in var_candidate:
                    for var_key, var_value in block_type_candidate.items():
                        code[var_key] = var_value
            else:
                # stage type
                for var_key, var_value in var_candidate.items():
                    code[var_key] = var_value
        # print(task_code)
        task_codes.append(code)
    extractor.set_code_list(task_codes)
    extractor.extract(block_level=block_level)
    print(f"Worker {worker_id} done")
    # print(extractor.op_dict)
    if not block_level:
        return extractor.op_dict
    else:
        return extractor.block_dict


def prebuild_worker(worker_id,
                    extractor,
                    worker_task,
                    task_code,
                    block_building=False,
                    block_level=False,
                    queue=None):
    extractor.clear_op_dict()
    num_tasks = len(worker_task)
    print(f"Worker {worker_id} created with {num_tasks} tasks")
    task_codes = []
    for i, stage_block_candidate in enumerate(worker_task):
        code = copy.deepcopy(task_code)
        for var_candidate in stage_block_candidate:
            if isinstance(var_candidate, tuple):
                # block type
                for block_type_candidate in var_candidate:
                    for var_key, var_value in block_type_candidate.items():
                        code[var_key] = var_value
            else:
                # stage type
                for var_key, var_value in var_candidate.items():
                    if var_key == "reduce_depth":
                        code[var_key] = [0 for _ in range(len(var_value))]
                    else:
                        code[var_key] = var_value
        # print(task_code)
        task_codes.append(code)
    extractor.set_code_list(task_codes)
    extractor.pre_build(block_building=block_building)

    if queue is not None:
        queue.put(extractor.built_op_dict)
        print(f"Worker {worker_id} done")
    else:
        return extractor.built_op_dict
    # if not block_level:
    #     return extractor.built_op_dict
    # else:
    #     return extractor.built_block_dict
