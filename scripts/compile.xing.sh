#Written by Barret Zoph, for questions email barretzoph@gmail.com
#Futher edit by Xing Shi (shixing19910105@gmail.com)

#Compilation script for compiling ZOPH_RNN
#The following 7 environmental variables must be set in order for compilation
#The default arguements are examples of what the paths should look like
#For the dependencies the following are required:
#-----------------------------------------------
#cuda version >= 8.0
#One of the following gcc versions >=  4.9.3
#CuDNN version = 4 
#Boost version = 1.51.0 or 1.55.0 
#Any version of Eigen

source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/gnu/gcc/4.9.3/setup.sh

PATH_TO_CUDA_INCLUDE=${PATH_TO_CUDA_INCLUDE:-"/usr/usc/cuda/8.0/include/"} 
PATH_TO_BOOST_INCLUDE=${PATH_TO_BOOST_INCLUDE:-"/usr/usc/boost/1.55.0/include/"}
PATH_TO_CUDA_LIB_64=${PATH_TO_CUDA_LIB_64:-"/usr/usc/cuda/8.0/lib64/"} 
PATH_TO_BOOST_LIB=${PATH_TO_BOOST_LIB:-"/usr/usc/boost/1.55.0/lib/"}
PATH_TO_CUDNN_V4_64=${PATH_TO_CUDNN_V4_64:-"/home/nlg-05/zoph/cudnn_v4/lib64/"}
PATH_TO_EIGEN=${PATH_TO_EIGEN:-"/home/nlg-05/zoph/eigen/"}
PATH_TO_CUDNN_INCLUDE=${PATH_TO_CUDNN_INCLUDE:-"/home/nlg-05/zoph/cudnn_v4/include/"}

#compile

#-DTIMER_DEBUG
nvcc -DCUDNN_STATIC -O3 -arch=sm_35 \
    -Xcompiler -fopenmp \
    -I $PATH_TO_CUDA_INCLUDE \
    -I $PATH_TO_BOOST_INCLUDE  \
    -I $PATH_TO_EIGEN \
    -I $PATH_TO_CUDNN_INCLUDE \
    -std=c++11 \
    -dc src/main.cu -o main.o

nvcc -c src/format.cc -o format.o

nvcc -arch=sm_35 -rdc=true -O3 main.o format.o -o ZOPH_RNN_XING\
    ${PATH_TO_BOOST_LIB}libboost_system.a \
    ${PATH_TO_BOOST_LIB}libboost_filesystem.a \
    ${PATH_TO_BOOST_LIB}libboost_program_options.a \
    ${PATH_TO_BOOST_LIB}libboost_regex.a \
    ${PATH_TO_CUDA_LIB_64}libcublas_static.a \
    ${PATH_TO_CUDA_LIB_64}libcudadevrt.a \
    ${PATH_TO_CUDA_LIB_64}libcudart_static.a \
    ${PATH_TO_CUDA_LIB_64}libculibos.a \
    ${PATH_TO_CUDA_LIB_64}libcurand_static.a \
    ${PATH_TO_CUDA_LIB_64}libcusolver_static.a \
    ${PATH_TO_CUDA_LIB_64}libcusparse_static.a \
    ${PATH_TO_CUDA_LIB_64}libnppc_static.a \
    ${PATH_TO_CUDA_LIB_64}libnppi_static.a \
    ${PATH_TO_CUDA_LIB_64}libnpps_static.a \
    ${PATH_TO_CUDNN_V4_64}libcudnn_static.a \
    

######         
