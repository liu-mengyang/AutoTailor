import os
import logging
import docker


engine_root = os.getenv("ENGINE_HOME")

class Engine:
    def __init__(
        self,
        type="ncnn",
        ncnn_vulkan=True,
        ncnn_arm_neon=True,
        ncnn_benchmark=True,
        ncnn_patch=None,
        extra_src_dir=None,
        tflite_plus_flex=False,
        tflite_hexagon_tool=False,
        tflite_patch=None,
        engine_option="benchncnn_custom"
    ) -> None:
        if type not in ["ncnn", "tflite", "ptmobile"]:
            raise ValueError("Invalid engine type")

        logging.info("Initializing engine: %s" % type)

        if not os.path.exists(engine_root):
            os.makedirs(engine_root)

        self.type = type
        self.ncnn_vulkan = ncnn_vulkan
        self.ncnn_arm_neon = ncnn_arm_neon
        self.ncnn_benchmark = ncnn_benchmark
        self.engine_root = engine_root
        self.ncnn_patch = ncnn_patch
        self.extra_src_dir = extra_src_dir
        self.tflite_plus_flex = tflite_plus_flex
        self.tflite_hexagon_tool = tflite_hexagon_tool
        self.tflite_patch = tflite_patch
        
        self.engine_name = self.__get_engine_name()
        if engine_option is not None:
            self.engine_name = engine_option

        self.engine_path = os.path.join(self.engine_root, self.engine_name)
        self.engine_option = engine_option
        if not os.path.exists(self.engine_path):
            logging.info("Building engine: %s" % type)
            self.__build_engine()

    def __get_command(self, engine_file_name="benchncnn_custom.cpp"):
        if self.type == "ncnn":
            if self.extra_src_dir:
                # for detecting kernel selection rule
                return f"""
                    export NCNN_ROOT=/home/src_dir \\
                    && mkdir -p $NCNN_ROOT/build \\
                    && cd $NCNN_ROOT/build \\
                    && cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \\
                        -DANDROID_ABI="arm64-v8a" \\
                        -DANDROID_PLATFORM=android-24 \\
                        {f"-DNCNN_VULKAN=ON" if self.ncnn_vulkan else ""} \\
                        {f"-DANDROID_ARM_NEON=ON" if self.ncnn_arm_neon else ""} \\
                        {f"-DNCNN_BENCHMARK=ON" if self.ncnn_benchmark else ""} \\
                        .. \\
                    && make -j$(nproc) \\
                    && make install \\
                    && mv $NCNN_ROOT/build/benchmark/benchncnn_kd_fp32 /home/out/benchncnn_kd_fp32 \\
                    && mv $NCNN_ROOT/build/benchmark/benchncnn_kd_fp16 /home/out/benchncnn_kd_fp16 \\
                    && mv $NCNN_ROOT/build/benchmark/benchncnn_kd_bf16 /home/out/benchncnn_kd_bf16 \\
                    && mv $NCNN_ROOT/build/benchmark/benchncnn_kd_int8 /home/out/benchncnn_kd_int8
                """
            elif self.ncnn_patch:
                return f"""
                    rm $NCNN_ROOT/benchmark/CMakeLists.txt \\
                    && cp /home/ncnn-patch/CMakeLists.txt $NCNN_ROOT/benchmark/CMakeLists.txt \\
                    && rm $NCNN_ROOT/benchmark/benchncnn_custom.cpp \\
                    && cp /home/ncnn-patch/{engine_file_name}.cpp $NCNN_ROOT/benchmark/benchncnn_custom.cpp \\
                    && mkdir /home/ncnn/build \\
                    && cd /home/ncnn/build \\
                    && cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \\
                        -DANDROID_ABI="arm64-v8a" \\
                        -DANDROID_PLATFORM=android-24 \\
                        {f"-DNCNN_VULKAN=ON" if self.ncnn_vulkan else ""} \\
                        {f"-DANDROID_ARM_NEON=ON" if self.ncnn_arm_neon else ""} \\
                        {f"-DNCNN_BENCHMARK=ON" if self.ncnn_benchmark else ""} \\
                        .. \\
                    && make -j$(nproc) \\
                    && make install \\
                    && mv /home/ncnn/build/benchmark/benchncnn_custom /home/out/{self.engine_name}
                """
            else:
                return f"""
                    mkdir /home/ncnn/build \\
                    && cd /home/ncnn/build \\
                    && cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \\
                        -DANDROID_ABI="arm64-v8a" \\
                        -DANDROID_PLATFORM=android-24 \\
                        {f"-DNCNN_VULKAN=ON" if self.ncnn_vulkan else ""} \\
                        {f"-DANDROID_ARM_NEON=ON" if self.ncnn_arm_neon else ""} \\
                        {f"-DNCNN_BENCHMARK=ON" if self.ncnn_benchmark else ""} \\
                        .. \\
                    && make -j$(nproc) \\
                    && make install \\
                    && mv /home/ncnn/build/benchmark/benchncnn_custom /home/out/{self.engine_name}
                """

        elif self.type == "tflite":
            if self.tflite_hexagon_tool and self.tflite_patch:
                return """
                    cp /home/tflite-patch/build_hexagon_tool.sh /home/build_hexagon_tool.sh\\
                    && bash /home/build_hexagon_tool.sh \\
                    && cd /home/tensorflow \\
                    && mv bazel-bin/tensorflow/lite/delegates/hexagon/hexagon_nn/libhexagon_interface.so /home/out/libhexagon_interface.so \\
                """
            elif self.tflite_plus_flex and self.tflite_patch:
                return f"""
                    bash /home/build_plus_flex.sh \\
                    && cd /home/tensorflow \\
                    && mv bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex /home/out/{self.engine_name}
                """
            else:
                return f"""
                    bash /home/build.sh \\
                    && cd /home/tensorflow \\
                    && mv bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /home/out/{self.engine_name}
                """
        elif self.type == "ptmobile":
            return f"""
                cd /home/pytorch \\
                && rm -rf build_android \\
                && BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON \\
                && mv build_android/bin/speed_benchmark_torch /home/out/{self.engine_name}
            """
        elif self.type == "ort":
            return f"""
                cd /home/onnxruntime \\
                && mv build/Android/Debug/onnxruntime_perf_test /home/out/{self.engine_name}
            """
        else:
            raise NotImplementedError

    def __get_engine_name(self):
        if self.type == "ncnn":
            if self.extra_src_dir:
                return "ncnn_kd"
            else:
                if self.ncnn_vulkan:
                    if self.ncnn_arm_neon:
                        if self.ncnn_benchmark:
                            return "ncnn_vulkan_arm_neon_benchmark"
                        else:
                            return "ncnn_vulkan_arm_neon"
                    else:
                        if self.ncnn_benchmark:
                            return "ncnn_vulkan_benchmark"
                        else:
                            return "ncnn_vulkan"
                else:
                    if self.ncnn_arm_neon:
                        if self.ncnn_benchmark:
                            return "ncnn_arm_neon_benchmark"
                        else:
                            return "ncnn_arm_neon"
                    else:
                        if self.ncnn_benchmark:
                            return "ncnn_benchmark"
                        else:
                            return "ncnn"
        elif self.type == "tflite":
            if self.tflite_plus_flex:
                return "tflite_plus_flex"
            else:
                return "tflite"
        elif self.type == "ptmobile":
            return "pytorchmobile"
        elif self.type == "ort":
            return "ort"
        else:
            raise NotImplementedError

    def __get_builder_image(self):
        if self.type == "ncnn":
            return "registry.cn-hangzhou.aliyuncs.com/lmy20/ncnn-engine-builder"
        elif self.type == "tflite":
            return "registry.cn-hangzhou.aliyuncs.com/lmy20/tflite-engine-builder"
        elif self.type == "ptmobile":
            return "registry.cn-hangzhou.aliyuncs.com/lmy20/ptmobile-engine-builder"
        else:
            raise NotImplementedError

    def __build_engine(self):
        client = docker.from_env()
        volumes = {self.engine_root: {"bind": "/home/out", "mode": "rw"}}
        if self.type == "ncnn" and self.ncnn_patch:
            volumes[self.ncnn_patch] = {"bind": "/home/ncnn-patch", "mode": "rw"}
        if self.type == "tflite" and self.tflite_patch:
            volumes[self.tflite_patch] = {"bind": "/home/tflite-patch", "mode": "rw"}
        print(self.__get_command(self.engine_option))
        logging.info("Command: %s" % self.__get_command(self.engine_option))

        if self.extra_src_dir:
            volumes[self.extra_src_dir] = {"bind": "/home/src_dir", "mode": "rw"}


        client.containers.run(
            image=self.__get_builder_image(),
            command="/bin/bash -c '%s'" % self.__get_command(self.engine_option),
            volumes=volumes,
            remove=True,
        )
    
    def build_ncnn_tool(self):
        client = docker.from_env()
        volumes = {self.engine_root: {"bind": "/home/out", "mode": "rw"}}
        if self.type == "ncnn" and self.ncnn_patch:
            volumes[self.ncnn_patch] = {"bind": "/home/ncnn-patch", "mode": "rw"}

        command =  f"""
                    export NCNN_ROOT=/home/ncnn \\
                    && mkdir -p $NCNN_ROOT/build \\
                    && cd $NCNN_ROOT/build \\
                    && cmake .. \\
                    && make -j$(nproc) \\
                    && make install \\
                    && mv $NCNN_ROOT/build/install/ /home/out/tool/
                    """


        client.containers.run(
            image=self.__get_builder_image(),
            command="/bin/bash -c '%s'" % command,
            volumes=volumes,
            remove=True,
        )
