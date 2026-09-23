#!/bin/bash

cd /home/onnxruntime

export ANDROID_NDK_API_LEVEL="30"
export ANDROID_BUILD_TOOLS_VERSION="31.0.0"
export ANDROID_SDK_API_LEVEL="30"

echo y | sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "emulator" \
  "platforms;android-${ANDROID_SDK_API_LEVEL}"

./build.sh --android \
  --android_sdk_path /android/sdk \
  --android_ndk_path /android/ndk \
  --android_abi arm64-v8a \
  --android_api 30 \
  --use_nnapi \
  --allow_running_as_root