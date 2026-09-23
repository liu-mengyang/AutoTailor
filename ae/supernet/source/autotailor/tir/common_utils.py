from .blocks import *    


def set_verbose_level(tir, verbose_level):
    def set_on_path(path, verbose_level):
        for block in path:
            if not isinstance(block, SingleOp):
                if block.paths is not None:
                    for path_key, block_path in block.paths.items():
                        set_on_path(block_path, verbose_level)
                elif block.flow is not None:
                    set_on_path(block.flow, verbose_level)
                else:
                    raise NotImplementedError
            else:
                block.verbose_level = verbose_level
    
    for i, stage in tir.stages.items():
        stage_path = []
        for block_id, block in stage.flow.items():
            stage_path.append(block)
        set_on_path(stage_path, stage.verbose_level)