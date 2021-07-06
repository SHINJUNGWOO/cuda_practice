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

__device__ void flat_conv(float* Input, float* Kernel, float* Output,int* image_size, int* kernel_size)
{
    //__shared__ float kernel_part[kernel_size[2]][kernel_size[3]][kernel_size[1]];
    __shared__ float kernel_part[3][3][3];
    int out_w = image_size[2]+2*PAD - kernel_size[3] + 1;

    int col_idx = blockIdx.x - PAD + threadIdx.x;
    int row_idx = blockIdx.y - PAD + threadIdx.y;
    int img_flat_size = image_size[1]*image_size[2];
    int kernel_flat_size = kernel_size[2]*kernel_size[3];
    if( image_size[2]>col_idx && col_idx >=0 && image_size[1]>row_idx && row_idx >=0)
    {
        kernel_part[threadIdx.y][threadIdx.x][threadIdx.z] = Input[(col_idx * image_size[2] +row_idx) + img_flat_size*threadIdx.z]
                                                           ;//* Kernel[threadIdx.y*kernel_size[3] + threadIdx.x + kernel_flat_size*threadIdx.z];
    }
    else
    {
        kernel_part[threadIdx.y][threadIdx.x][threadIdx.z] = 0;
    }
    __syncthreads;

    atomicAdd(&(Output[blockIdx.x * out_w +blockIdx.y]), kernel_part[threadIdx.y][threadIdx.x][threadIdx.z]);
}


__global__ void conv(float* Input, float* Kernel, float* Output,int* image_size, int* kernel_size)
{   

    int flat_img_size = kernel_size[3]*kernel_size[2]*kernel_size[1];
    flat_conv(Input, Kernel, Output, image_size, kernel_size);
}


__host__ int main(void)
{

    float h_a[3][64][64] ={0.0};
    h_a[0][0][0] = 1.1;
    h_a[1][0][0] = 1.1;
    h_a[2][0][0] = 1.1;
    float h_b[1][3][3][3] ={ 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};

    float h_c[1][64][64] ={0.0};
    int kernel_size[4] ={1,3,3,3}; //O I H W;
    int image_size[3] = {3,64,64}; //  O H W;


    float *cimg;
    float *coimg;
    float *ckernel;
    int * cimg_size;
    int * ckernel_size;
    cudaMalloc((void***)&cimg,sizeof(h_a));
    cudaMalloc((void***)&ckernel,sizeof(h_b));
    cudaMalloc((void***)&coimg,sizeof(h_c));
    cudaMalloc(&cimg_size,sizeof(image_size));
    cudaMalloc(&ckernel_size,sizeof(kernel_size));


    cudaMemcpy(cimg,h_a,sizeof(h_a),cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel,h_b,sizeof(h_b),cudaMemcpyHostToDevice);
    cudaMemcpy(cimg_size,image_size,sizeof(image_size),cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel_size,kernel_size,sizeof(kernel_size),cudaMemcpyHostToDevice);

    dim3 threads(kernel_size[3], kernel_size[2], kernel_size[1]);
	dim3 grid(image_size[2],image_size[1],1);
    clock_t start = clock(); 

    conv <<< grid,threads>>>(cimg,ckernel,coimg,cimg_size,ckernel_size);
    //Convolution <<< grid,threads>>>(cimg,ckernel,coimg,cimg_size,ckernel_size);

    clock_t end = clock();
    cudaMemcpy(h_c,coimg,sizeof(h_c),cudaMemcpyDeviceToHost);

    for(int j =0; j < WB;j ++)
    {
        for(int k =0; k < WB;k ++)
        {
            printf("%.0f ",h_c[0][k][j]);
        }
        printf("\n");   
    }
    printf("\n");


    printf("%f",(float)(end - start)/CLOCKS_PER_SEC);
}