from tqdm import tqdm
from time import time


class Extractor(object):
    def __init__(self, tailor):
        self.tailor = tailor
        self.global_vars = tailor.global_vars
        self.stage_vars = tailor.stage_vars
        self.block_vars = tailor.block_vars

        self.supercode = tailor.supercode

        self.op_dict = {}
        self.block_dict = {}
        self.built_op_dict = {}

        self.code_list = []

    def set_code_list(self, code_list):
        self.code_list = code_list

    def clear_op_dict(self):
        self.op_dict = {}

    def pre_build(self, block_building=False):
        for code in self.code_list:
            self.tailor.transform(code)
            self.tailor.tir.pre_build(
                self.built_op_dict,
                block_building=block_building)
            # for op_type in self.built_op_dict.keys():
            #     print(f"Length of {op_type}: {len(self.built_op_dict[op_type])}")
            # print("---------------------------")

    def extract(self, block_level=False, drop_dup=True):
        for code in self.code_list:
            # st = time()
            self.tailor.transform(code)
            # ed = time()
            # transform_time = ed - st
            if not block_level:
                # st = time()
                try:
                    op_dict = self.tailor.tir.get_ops(drop_dup=drop_dup)
                except AssertionError as e:
                    print(code)
                    raise e
                # ed = time()
                # get_time = ed - st
                # st = time()
                for op_type, sub_op_set in op_dict.items():
                    if op_type in self.op_dict:
                        if drop_dup:
                            self.op_dict[op_type] = self.op_dict[op_type] | sub_op_set
                        else:
                            # set is list actually
                            self.op_dict[op_type] += sub_op_set
                        # if op_type == "Reshape":
                        #     print(self.op_dict[op_type])
                        #     print(type(list(self.op_dict[op_type])[0][-1]))
                        # if op_type == "MatMul":
                        #     print(self.op_dict[op_type])
                    else:
                        self.op_dict[op_type] = sub_op_set # set of tuples
                    # if op_type == "Mul":
                        # print(sub_op_set)
                        # if ((1, 121, 480), (1, 121, 480), 480, 480) in sub_op_set:
                        #     print(f"Meet: {code}")
                # ed = time()
                # add_time = ed - st
                # print(f"Transform time: {transform_time}; Get time: {get_time}; Add time: {add_time}")
            else:
                # block-level extracting
                # FIXME: not support drop_dup=False
                block_dict = self.tailor.tir.get_blocks()
                for block_type, sub_block_set in block_dict.items():
                    if block_type in self.block_dict:
                        self.block_dict[block_type] = self.block_dict[block_type] | sub_block_set
                    else:
                        self.block_dict[block_type] = sub_block_set
