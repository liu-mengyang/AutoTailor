from .engine import Engine
import os

class EngineBuilder:
    def build_for_ncnn(self):
        ncnn_patch_dir = os.path.join(os.getenv("AUTOTAILOR_HOME"), "infra/engine_builder/docker/ncnn-patch")
        ncnn_engine = Engine(type="ncnn",
                             ncnn_patch=ncnn_patch_dir)
    
    def build_for_ncnn_tool(self):
        ncnn_engine = Engine(type="ncnn")
        ncnn_engine.build_ncnn_tool()

    def build_for_tflite(self):
        tflite_engine = Engine(type="tflite")
        

    def build_for_ncnn_options(self, engine_option_ncnn):
        ncnn_patch_dir = os.path.join(os.getenv("AUTOTAILOR_HOME"), "infra/engine_builder/docker/ncnn-patch")
        ncnn_engine = Engine(type="ncnn",
                             ncnn_vulkan=True,
                             ncnn_patch=ncnn_patch_dir,
                             engine_option=engine_option_ncnn)
    
    def build_for_tflite_options(self):
        tflite_patch_dir = os.path.join(os.getenv("AUTOTAILOR_HOME"), "infra/engine_builder/docker/tflite-patch")
        tflite_engine = Engine(type="tflite",
                               tflite_plus_flex=True,
                               tflite_patch=tflite_patch_dir)
        tflite_hexagon_tool = Engine(type="tflite",
                                     tflite_hexagon_tool=True,
                                     tflite_patch=tflite_patch_dir)

    def build_for_kernel_selection_rule_detection(self, extra_dir):
        ncnn_engine = Engine(type="ncnn",
                             ncnn_benchmark=False,
                             extra_src_dir=extra_dir)

    def build_for_ptmobile(self):
        ptmobile_engine = Engine(type="ptmobile")
    
    def build_for_ort(self):
        ptmobile_engine = Engine(type="ort")

