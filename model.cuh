#include <cuda.h>
#include <cuda_runtime.h>

float* convolution_relu(float* Input, float* Kernel, int* h_image_size, int* h_kernel_size, int h_pad,int h_stride);
float* maxpooling(float* Input, int* h_image_size, int h_kernel_size);
float* fully_connected(float* Input,float* Kernel,int* kernel_size);