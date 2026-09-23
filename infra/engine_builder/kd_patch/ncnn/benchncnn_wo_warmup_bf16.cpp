// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <numeric>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 0;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working"
#else
#define MODEL_DIR "./ncnn_models/"
#endif

    char parampath[256];
    char binpath[256];
    sprintf(parampath, "%s.param", comment);
    sprintf(binpath, "%s.bin", comment);
    net.load_param(parampath);
    net.load_model(binpath);

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    ncnn::Mat out;

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    double time_var = 0;
    std::vector<double> time_vec;
    std::vector<double> time_diff;

    double timestamp_start = ncnn::get_current_time();

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
        }

        double end = ncnn::get_current_time();
        double time = end - start;

        time_vec.push_back(time);

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
    }

    time_avg = std::accumulate(time_vec.begin(), time_vec.end(), 0.0) / time_vec.size();

    for(int i = 0; i < time_vec.size(); i++)
        time_diff.push_back(pow(time_vec[i] - time_avg, 2));
    time_var = sqrt(std::accumulate(time_diff.begin(), time_diff.end(), 0.0) / time_diff.size());


    double timestamp_end = ncnn::get_current_time();

    fprintf(stdout, "time_min %.2f\n", time_min);
    fprintf(stdout, "time_max %.2f\n", time_max);
    fprintf(stdout, "time_avg %.2f\n", time_avg);
    fprintf(stdout, "time_var %.2f\n", time_var);
    
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 1;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

    char *model_name = argv[6];
    int model_shape_x = atoi(argv[7]);
    int model_shape_y = atoi(argv[8]);
    int model_shape_z = atoi(argv[9]);

#ifdef __EMSCRIPTEN__
    EM_ASM(
        FS.mkdir('/working');
        FS.mount(NODEFS, {root: '.'}, '/working'););
#endif // __EMSCRIPTEN__

    bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 0;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    // default option
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = false;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = true;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    // run
    // benchmark("vgg11", ncnn::Mat(224, 224, 3), opt);
    // benchmark("vgg13", ncnn::Mat(224, 224, 3), opt);
    // benchmark("vgg16", ncnn::Mat(224, 224, 3), opt);
    // benchmark("vgg19", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnet18", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnet34", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnet50", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnet101", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnet152", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnext50_32x4d", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnext101_32x4d", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnext101_32x8d", ncnn::Mat(224, 224, 3), opt);
    // benchmark("resnext152_32x4d", ncnn::Mat(224, 224, 3), opt);
    // benchmark("efficientnet_b0", ncnn::Mat(224, 224, 3), opt);
    // benchmark("efficientnet_b1", ncnn::Mat(240, 240, 3), opt);
    // benchmark("efficientnet_b2", ncnn::Mat(260, 260, 3), opt);
    // benchmark("efficientnet_b3", ncnn::Mat(280, 280, 3), opt);
    // benchmark("efficientnet_b4", ncnn::Mat(380, 380, 3), opt);
    // benchmark("efficientnet_b5", ncnn::Mat(456, 456, 3), opt);
    // benchmark("efficientnet_b6", ncnn::Mat(528, 528, 3), opt);
    // benchmark("efficientnet_b7", ncnn::Mat(600, 600, 3), opt);
    // benchmark("efficientnet_b8", ncnn::Mat(672, 672, 3), opt);
    benchmark(model_name, ncnn::Mat(model_shape_x, model_shape_y, model_shape_z), opt);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
