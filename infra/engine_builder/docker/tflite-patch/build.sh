#!/bin/bash

cd /home/tensorflow

export TF_PYTHON_VERSION="3.10"
export ANDROID_NDK_API_LEVEL="26"
export ANDROID_BUILD_TOOLS_VERSION="30.0.3"
export ANDROID_SDK_API_LEVEL="30"

echo y | sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_SDK_API_LEVEL}"

configs=(
  '/usr/bin/python3'
  '/usr/lib/python3/dist-packages'
  'N'
  'N'
  'N'
  'N'
  '-march=native -Wno-sign-compare -Wno-c++20-designator -Wno-gnu-inline-cpp-without-extern'
  'y'
  '/android/sdk'
)

printf '%s\n' "${configs[@]}" | ./configure
# Configure Bazel.
source tensorflow/tools/ci_build/release/common.sh
install_bazelisk

bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark:benchmark_model \
  --verbose_failures
