#export PATH=$PATH:/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/
#CXX=clang++ CC=clang scons Werror=1 -j4 debug=0 asserts=0 neon=1 opencl=1 embed_kernels=1 os=android arch=arm64-v8a examples=0

export PATH=$PATH:/media/luyao/video_send_back/install_package/arm-linux-android-ndk-r17b/bin/

CXX=clang++ CC=clang scons Werror=1 -j4 debug=0 asserts=0 neon=1 opencl=1 embed_kernels=1 os=android arch=armv7a examples=0
#cppthreads=0
