#!/usr/bin/env bash
set -euo pipefail
curl -fL --retry 3 https://dl.google.com/android/repository/android-ndk-r25c-linux.zip -o /tmp/ndk.zip
unzip -q /tmp/ndk.zip -d /opt
rm /tmp/ndk.zip
git clone --depth 1 --branch 20240102 https://github.com/Tencent/ncnn.git /opt/ncnn
cp /src/benchncnn_custom.cpp /opt/ncnn/benchmark/
cp /src/CMakeLists.txt /opt/ncnn/benchmark/
cmake -S /opt/ncnn -B /opt/ncnn/build-android -G Ninja  -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r25c/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release  -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=ON
cmake --build /opt/ncnn/build-android --target benchncnn_custom --parallel 4
mkdir -p /out
cp /opt/ncnn/build-android/benchmark/benchncnn_custom /out/
git -C /opt/ncnn rev-parse HEAD > /out/ncnn-commit.txt
sha256sum /out/benchncnn_custom > /out/SHA256SUMS
