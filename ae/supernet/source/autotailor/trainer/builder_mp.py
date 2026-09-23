from torch.multiprocessing import Process


class BuildWorker(Process):
    def __init__(self,
                 tailor,
                 subnet_configs,
                 branch_configs):
        self.tailor = tailor
        self.subnet_configs = subnet_configs
        self.branch_configs = branch_configs
    
    def run(self):
        
        for epoch, epoch_dict in self.branch_samples.items():
            for batch, branch_sample in epoch_dict.items():
                backbone_config = branch_sample["shared_backbone"]
                branch_config = branch_sample["branches"]
        
                dynamic_batch_size = len(branch_config)
                for i in range(dynamic_batch_size):
                    next_model = self.tailor.tir.build(
                        branching=True,
                        branch_stage_id=backbone_config["branch_stage_id"],
                        branch_block_id=backbone_config["branch_block_id"],
                        branching_id=i,
                        num_heads=len(branch_config),
                        half_end=True
                    )