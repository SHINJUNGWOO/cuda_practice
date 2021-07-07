#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 32
#define WA 64   
#define HA 64     
#define HC 3     
#define WC 3
#define PAD 1
#define WB (WA+2*PAD - WC + 1)
#define HB (HA+2*PAD - HC + 1)
#define CHANNEL_SIZE 3

__constant__  int image_size[3];
__constant__  int kernel_size[4];
__constant__  int pad[1];
__constant__  int stride[1];


__device__ void flat_conv(float* Input, float* Kernel, float* Output,/* int* image_size, int* kernel_size, int* pad,int* stride,*/int* out_w)
{
    //__shared__ float kernel_part[kernel_size[2]][kernel_size[3]][kernel_size[1]];
    //__shared__ float kernel_part[3][3][3];
    extern __shared__ float kernel_part[];
    

    int col_idx = stride[0]*blockIdx.x - pad[0] + threadIdx.x;
    int row_idx = stride[0]*blockIdx.y - pad[0] + threadIdx.y;
    int img_flat_size = image_size[1]*image_size[2];
    int kernel_flat_size = kernel_size[2]*kernel_size[3];
    if( image_size[2]>col_idx && col_idx >=0 && image_size[1]>row_idx && row_idx >=0)
    {
        kernel_part[(threadIdx.y * kernel_size[3]+threadIdx.x)*kernel_size[1]+threadIdx.z] 
            = Input[(col_idx * image_size[2] +row_idx) + img_flat_size*threadIdx.z]
            * Kernel[threadIdx.y*kernel_size[3] + threadIdx.x + kernel_flat_size*threadIdx.z];
    }
    else
    {
        kernel_part[(threadIdx.y * kernel_size[3]+threadIdx.x)*kernel_size[1]+threadIdx.z] = 0;
    }
    //__syncthreads;

    atomicAdd(&(Output[blockIdx.x * out_w[0] +blockIdx.y]), kernel_part[(threadIdx.y * kernel_size[3]+threadIdx.x)*kernel_size[1]+threadIdx.z]);
}

__global__ void conv(float* Input, float* Kernel, float* Output/*, int* image_size, int* kernel_size, int* pad,int* stride*/)
{   
    int out_w = (image_size[2]+2*pad[0] - kernel_size[3])/stride[0] + 1;
    int out_h = (image_size[1]+2*pad[0] - kernel_size[2])/stride[0] + 1;
    int flat_kernel_size = kernel_size[3]*kernel_size[2]*kernel_size[1];
    int flat_img_size = out_w*out_h;
    flat_conv(Input, Kernel + flat_kernel_size*blockIdx.z , Output + flat_img_size*blockIdx.z, /* image_size, kernel_size, pad, stride,*/ &out_w);

}

__global__ void relu(float*Input)
{
    int col_idx = blockIdx.x* blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y* blockDim.y + threadIdx.y;
    if ( (Input + blockIdx.z * blockDim.x*blockDim.y)[row_idx*gridDim.x*blockDim.x + col_idx]<0)
    {
        (Input + blockIdx.z * blockDim.x*blockDim.y)[row_idx*gridDim.x*blockDim.x + col_idx] = 0;
    }
}


__host__ float* convolution_relu(float* Input, float* Kernel, int* h_image_size, int* h_kernel_size, int h_pad,int h_stride)
{
    int out_w = (h_image_size[2]+2*h_pad - h_kernel_size[3])/h_stride + 1;
    int out_h = (h_image_size[1]+2*h_pad - h_kernel_size[2])/h_stride + 1;
    int flat_kernel_size = h_kernel_size[3]* h_kernel_size[2]* h_kernel_size[1]*sizeof(float);
    float* Output;
    
    cudaMemcpyToSymbol(image_size,h_image_size,sizeof(int)*3);
    cudaMemcpyToSymbol(kernel_size,h_kernel_size,sizeof(int)*4);
    cudaMemcpyToSymbol(pad,&h_pad,sizeof(int));
    cudaMemcpyToSymbol(stride,&h_stride,sizeof(int));
    cudaMalloc((void***)&Output,out_w*out_h*h_kernel_size[0]*sizeof(float));
    
    dim3 threads_c(h_kernel_size[3], h_kernel_size[2], h_kernel_size[1]);
	dim3 grid_c(out_w,out_h,h_kernel_size[0]);

    conv <<< grid_c,threads_c,flat_kernel_size>>>(Input,Kernel,Output);

    dim3 threads_r(32,32);
	dim3 grid_r(out_w/32,out_h/32,h_kernel_size[0]);
    relu <<<grid_r,threads_r >>>(coimg);
    
    return Output;
}


        // dim3 threads_r(32,32);
	// dim3 grid_r(out_size[2]/32,out_size[1]/32,out_size[0]);
    // relu <<<grid_r,threads_r >>>(coimg);
}

//// HOST /////
void randomInit(float* data, int size)
{
    
	for (int i = 0; i < size; ++i)
		data[i] = (rand() / (float)RAND_MAX) +0.5;
}
__host__ int main(void)
{

    int h_kernel_size[4] ={2,3,3,3}; //O I H W;
    int h_image_size[3] = {3,64,64}; //  O H W;

    float* h_a; float *h_b; float* h_c;

    int h_a_size = sizeof(float)*3*64*64;
    int h_b_size = sizeof(float)*2*3*3*3;
    int h_c_size = sizeof(float)*2*32*32;

    h_a = (float*)malloc(h_a_size);
    h_b = (float*)malloc(h_b_size);
    h_c = (float*)malloc(h_c_size);
    
    randomInit(h_a,3*64*64);
    randomInit(h_b,2*3*3*3);
    int h_pad = 1;
    int h_stride = 2;



    float *cimg;
    float *coimg;
    float *ckernel;
    cudaMalloc((void***)&cimg,h_a_size);
    cudaMalloc((void***)&ckernel,h_b_size);

    cudaMemcpy(cimg,h_a,h_a_size,cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel,h_b,h_b_size,cudaMemcpyHostToDevice);

    clock_t start = clock(); 
    coimg = convolution_relu(cimg,ckernel,h_image_size,h_kernel_size,h_pad,h_stride);

    clock_t end = clock();
    cudaMemcpy(h_c,coimg,h_c_size,cudaMemcpyDeviceToHost);

    int cnt = 0;
    for(int i = 0;i < 2; i++)
    {
        for(int j =0; j < 32;j ++)
        {
            for(int k =0; k < 32;k ++)
            {
                //printf("%.0f ",h_c[cnt]);
                printf("%.0f ",h_c[cnt]);
                cnt +=1;
            }
            printf("\n");   
        }
        printf("\n");
    }

    cudaFree(cimg);
    cudaFree(ckernel);
    cudaFree(coimg);
    //cudaFree(cimg_size);
    //cudaFree(ckernel_size);
    //cudaFree(cpad);
    //cudaFree(cstride);
    
    
    printf("%f",(float)(end - start)/CLOCKS_PER_SEC);
}