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

__device__ void flat_conv(float* Input, float* Kernel, float* Output,int* image_size, int* kernel_size, int* pad,int* stride,int* out_w)
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

__global__ void relu(float*Input, float*Output)
{
    int col_idx = blockIdx.x* blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y* blockDim.y + threadIdx.y;
    if ( (Input + blockIdx.z * blockDim.x*blockDim.y)[row_idx*gridDim.x*blockDim.x + col_idx]<0)
    {
        (Output + blockIdx.z * blockDim.x*blockDim.y)[row_idx*gridDim.x*blockDim.x + col_idx] = 0;
    }
}


__global__ void conv(float* Input, float* Kernel, float* Output,int* image_size, int* kernel_size,int* pad,int* stride)
{   
    int out_w = (image_size[2]+2*pad[0] - kernel_size[3])/stride[0] + 1;
    int out_h = (image_size[1]+2*pad[0] - kernel_size[2])/stride[0] + 1;
    int flat_kernel_size = kernel_size[3]*kernel_size[2]*kernel_size[1];
    int flat_img_size = out_w*out_h;
    flat_conv(Input, Kernel + flat_kernel_size*blockIdx.z , Output + flat_img_size*blockIdx.z, image_size, kernel_size, pad, stride, &out_w);
}

void randomInit(float* data, int size)
{
    
	for (int i = 0; i < size; ++i)
		data[i] = (rand() / (float)RAND_MAX) -0.5;
}
__host__ int main(void)
{

    // float h_a[3][64][64] ={0.0};
    // h_a[0][0][0] = 2.1;
    // h_a[1][0][0] = 2.1;
    // h_a[2][0][0] = 2.1;
    // float h_b[2][3][3][3] ={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    //                         1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    //                         1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    //                         2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    //                         2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    //                         2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};

    // float h_c[2][64][64] ={0.0};
    int kernel_size[4] ={2,3,3,3}; //O I H W;
    int image_size[3] = {3,64,64}; //  O H W;
    int out_size[3] = {2,32,32};

    float* h_a; float *h_b; float* h_c;

    int h_a_size = sizeof(float)*3*64*64;
    int h_b_size = sizeof(float)*2*3*3*3;
    int h_c_size = sizeof(float)*2*32*32;

    h_a = (float*)malloc(h_a_size);
    h_b = (float*)malloc(h_b_size);
    h_c = (float*)malloc(h_c_size);
    
    randomInit(h_a,3*64*64);
    randomInit(h_b,2*3*3*3);
    int pad = 1;
    int stride = 2;

    float *cimg;
    float *coimg;
    float *ckernel;
    int * cimg_size;
    int * ckernel_size;
    int * cpad;
    int * cstride;
    cudaMalloc((void***)&cimg,h_a_size);
    cudaMalloc((void***)&ckernel,h_b_size);
    cudaMalloc((void***)&coimg,h_c_size);
    cudaMalloc(&cimg_size,sizeof(image_size));
    cudaMalloc(&ckernel_size,sizeof(kernel_size));
    cudaMalloc(&cpad,sizeof(int));
    cudaMalloc(&cstride,sizeof(int));

    cudaMemcpy(cimg,h_a,h_a_size,cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel,h_b,h_b_size,cudaMemcpyHostToDevice);
    cudaMemcpy(cimg_size,image_size,sizeof(image_size),cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel_size,kernel_size,sizeof(kernel_size),cudaMemcpyHostToDevice);
    cudaMemcpy(cpad,&pad,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(cstride,&stride,sizeof(int),cudaMemcpyHostToDevice);

    dim3 threads(kernel_size[3], kernel_size[2], kernel_size[1]);
	dim3 grid(out_size[2],out_size[1],out_size[0]);
    clock_t start = clock(); 


    int flat_kernel_size = kernel_size[3]* kernel_size[2]* kernel_size[1]*sizeof(float);
    conv <<< grid,threads,flat_kernel_size>>>(cimg,ckernel,coimg,cimg_size,ckernel_size,cpad,cstride);
    dim3 threads_r(32,32);
	dim3 grid_r(out_size[2]/32,out_size[1]/32,out_size[0]);
    relu <<<grid_r,threads_r >>>(coimg,coimg);
    //Convolution <<< grid,threads>>>(cimg,ckernel,coimg,cimg_size,ckernel_size);

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
    cudaFree(cimg_size);
    cudaFree(ckernel_size);
    cudaFree(cpad);
    cudaFree(cstride);
    
    
    printf("%f",(float)(end - start)/CLOCKS_PER_SEC);
}