module load cuda/7.0
module load gcc/4.7.1
module load boost/1.51.0

nvcc -O3 -g  -I /opt/apps/cuda/7.0/include/ -I /opt/apps/gcc4_7/boost/1.51.0/include/  /opt/apps/gcc4_7/boost/1.51.0/lib/libboost_system.a      /opt/apps/gcc4_7/boost/1.51.0/lib/libboost_filesystem.a  /opt/apps/gcc4_7/boost/1.51.0/lib/libboost_program_options.a -I /home/03635/tg829350/eigen/  -std=c++11 -lcublas -lcurand  main.cu