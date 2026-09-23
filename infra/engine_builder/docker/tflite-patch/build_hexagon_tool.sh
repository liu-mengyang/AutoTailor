#!/bin/bash

cd /home/tensorflow

export TF_PYTHON_VERSION="3.10"
export ANDROID_NDK_API_LEVEL="26"
export ANDROID_BUILD_TOOLS_VERSION="30.0.3"
export ANDROID_SDK_API_LEVEL="30"

bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so \
  --verbose_failures
