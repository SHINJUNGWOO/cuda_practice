#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define IMW 64   
#define IMH 64     
#define IMC 3 
 
#define WO 3
#define WI 3
#define WH 3
#define WW 3

#define PAD 1
#define STIRDE 2

#define OIMW ((IMW+2*PAD-WW)/STIRDE+1) 
#define OIMH ((IMH+2*PAD-WH)/STIRDE+1) 

__constant__  int image_size[3];
__constant__  int kernel_size[4];
__constant__  int pad[1];
__constant__  int stride[1];


__device__ void flat_conv(float* Input, float* Kernel, float* Output,/* int* image_size, int* kernel_size, int* pad,int* stride,*/int* out_w)
{
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
    relu <<<grid_r,threads_r >>>(Output);
    
    //cudaFree(Input);
    //cudaFree(Kernel);
    return Output;
}

__global__ void g_maxpooling(float* Input,float* Output)
{
    
    int row_idx = blockIdx.y*blockDim.x +threadIdx.x;
    int col_len = gridDim.x*blockDim.x;
    int row_len = gridDim.y*blockDim.x;
    int flat_img_size = col_len*row_len;
    int bleft_start_idx = row_idx*col_len + blockIdx.x*blockDim.x +flat_img_size*threadIdx.y;
    extern __shared__ float kernel_part[];
    //size of double of kernel
    kernel_part[2*threadIdx.y+ threadIdx.x] =  Input[bleft_start_idx] >
                                               Input[bleft_start_idx+1] ?
                                               Input[bleft_start_idx]:
                                               Input[bleft_start_idx+1];
    __syncthreads();
    if(threadIdx.x ==0)
    {
        Output[blockIdx.y*gridDim.x+blockIdx.x + threadIdx.y*gridDim.x*gridDim.y] = kernel_part[2*threadIdx.y]> kernel_part[2*threadIdx.y+1]?
                                                                          kernel_part[2*threadIdx.y]: kernel_part[2*threadIdx.y+1];
    }
    __syncthreads();
}
__host__ float* maxpooling(float* Input, int* h_image_size, int h_kernel_size)
{
    float* Output;
    int col_len = h_image_size[2]/h_kernel_size;
    int row_len = h_image_size[1]/h_kernel_size;
    cudaMalloc((void***)&Output,row_len*col_len*h_image_size[0]*sizeof(float));
    dim3 threads(2,h_image_size[0]);
	dim3 grid(col_len,row_len);
    int shm_len = 2*h_image_size[0]*sizeof(float);

    g_maxpooling<<<grid,threads,shm_len>>>(Input,Output);
    return Output;

}


//// HOST /////
void randomInit(float* data, int size)
{
    
	for (int i = 0; i < size; ++i)
		data[i] =1;// (rand() / (float)RAND_MAX) +0.5;
}
__host__ int main(void)
{

    int h_kernel_size[4] ={WO,WI,WH,WW}; //O I H W;
    int h_image_size[3] = {IMC,IMH,IMW}; //  O H W;

    float* h_img; float *h_kernel; float* h_out;

    int h_img_len = sizeof(float)*IMC*IMH*IMW;
    int h_kernel_len = sizeof(float)*WO*WI*WH*WW;
    int h_out_len = sizeof(float)*WO*OIMH*OIMW;

    h_img = (float*)malloc(h_img_len);
    h_kernel = (float*)malloc(h_kernel_len);
    h_out = (float*)malloc(h_out_len);
    
    h_img[400] = 1;
    h_img[400+IMH*IMW] =1;
    h_img[400+2*IMH*IMW] =1;
    //randomInit(h_img,h_img_len/sizeof(float));
    randomInit(h_kernel,h_kernel_len/sizeof(float));
    int h_pad = PAD;
    int h_stride = STIRDE;



    float *cimg;
    
    float *ckernel;
    cudaMalloc((void***)&cimg,h_img_len);
    cudaMalloc((void***)&ckernel,h_kernel_len);

    cudaMemcpy(cimg,h_img,h_img_len,cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel,h_kernel,h_kernel_len,cudaMemcpyHostToDevice);

    clock_t start = clock();
    float *coimg_1;
    coimg_1 = convolution_relu(cimg,ckernel,h_image_size,h_kernel_size,1,1);
    coimg_1 = maxpooling(cimg,h_image_size,2);
    h_image_size[0] = 3;h_image_size[1] = 32;h_image_size[2] = 32;
    float *coimg;
    coimg = convolution_relu(coimg_1,ckernel,h_image_size,h_kernel_size,1,1);
    clock_t end = clock();
    cudaMemcpy(h_out,coimg,h_out_len,cudaMemcpyDeviceToHost);

    int cnt = 0;
    for(int i = 0;i < WO; i++)
    {
        for(int j =0; j < OIMH;j ++)
        {
            for(int k =0; k < OIMW;k ++)
            {
                //printf("%.0f ",h_c[cnt]);
                printf("%.1f ",h_out[cnt]);
                cnt +=1;
            }
            printf("\n");   
        }
        printf("\n");
    }

    cudaFree(cimg);
    cudaFree(ckernel);
    cudaFree(coimg_1);
    //cudaFree(cimg_size);
    //cudaFree(ckernel_size);
    //cudaFree(cpad);
    //cudaFree(cstride);
    
    
    printf("%f",(float)(end - start)/CLOCKS_PER_SEC);
}