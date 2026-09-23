#!/bin/bash

cd /home/tensorflow

export TF_PYTHON_VERSION="3.10"
export ANDROID_NDK_API_LEVEL="26"
export ANDROID_BUILD_TOOLS_VERSION="30.0.3"
export ANDROID_SDK_API_LEVEL="30"

echo yes | sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_SDK_API_LEVEL}"

bazel build -c opt \
  --config=android_arm64 \
  --config=monolithic \
  tensorflow/lite/tools/benchmark:benchmark_model_plus_flex \
  --verbose_failures
