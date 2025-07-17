# export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/include:$CPLUS_INCLUDE_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
# export CPLUS_INCLUDE_PATH=$SpInfer_HOME/third_party/sputnik/include:$SpInfer_HOME/third_party/glog/build/include:/usr/local/cuda/include
export CPLUS_INCLUDE_PATH=$SpInfer_HOME/third_party/glog/build/include:/usr/local/cuda/include
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:$CPLUS_INCLUDE_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATHexport CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH

cd ${SpInfer_HOME}/third_party/glog  && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${SpInfer_HOME}/third_party/glog/build -DCMAKE_BUILD_TYPE=Release ..
make -j
make install 


GlogPath="${SpInfer_HOME}/third_party/glog"
if [ -z "$GlogPath" ]
then
  echo "Defining the GLOG path is necessary, but it has not been defined."
else
  export GLOG_PATH=${SpInfer_HOME}/third_party/glog
  export CPLUS_INCLUDE_PATH=$GLOG_PATH/build/include:/usr/local/cuda/include
  export LD_LIBRARY_PATH=$GLOG_PATH/build/lib:$LD_LIBRARY_PATH
  export LIBRARY_PATH=$GLOG_PATH/build/lib:$LIBRARY_PATH
fi



cd ${SpInfer_HOME}/third_party/sputnik  && mkdir -p build && cd build
cmake .. -DGLOG_INCLUDE_DIR=$GLOG_PATH/build/include -DGLOG_LIBRARY=$GLOG_PATH/build/lib/libglog.so -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DCUDA_ARCHS="70" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_STANDARD=14 -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CXX_FLAGS="-I$GLOG_PATH/build/include -I/usr/local/cuda/include" -DCMAKE_C_FLAGS="-I$GLOG_PATH/build/include"
make -j12 

# SputnikPath="${SpInfer_HOME}/third_party/sputnik"
# if [ -z "$SputnikPath" ]
# then
#   echo "Defining the Sputnik path is necessary, but it has not been defined."
# else
#   export SPUTNIK_PATH=$SputnikPath
#   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SPUTNIK_PATH/build/sputnik
# fi

