
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__constant__  int image_size[3];
__constant__  int kernel_size[4];
__constant__  int stride[1];


__device__ void flat_conv(float* Input, float* Kernel, float* Output,int out_w)
{
    extern __shared__ float kernel_part[];

    int out_col_idx = (blockIdx.x*blockDim.x + threadIdx.x);
    int out_row_idx = (blockIdx.y*blockDim.y + threadIdx.y);

    int col_idx = stride[0]*out_col_idx;
    int row_idx = stride[0]*out_row_idx;

    int tmp =0;
    for(int i =-1; i < 2 ; i ++)
    {
        for(int j = -1; j<2 ; j++)
        {
            if ((col_idx+i) >= 0 && (row_idx+j) >= 0 && (col_idx+i) < image_size[2] && (row_idx+j) < image_size[1])
            {
                tmp +=
                Input[ (col_idx+i) * image_size[2] + row_idx+j + image_size[0]*threadIdx.z] *
                Kernel[ (i+1)*kernel_size[3] +(j+1) + kernel_size[1]*threadIdx.z];
            }

        }
    }
    kernel_part[threadIdx.y*blockDim.x + threadIdx.x + kernel_size[1]*threadIdx.z] = tmp;

    atomicAdd(&(Output[out_col_idx* out_w +out_row_idx]),kernel_part[threadIdx.y*blockDim.x + threadIdx.x + kernel_size[1]*threadIdx.z]);


}

__global__ void conv(float* Input, float* Kernel, float* Output)
{   
    int out_w = (image_size[2]+2*1 - kernel_size[3])/stride[0] + 1;
    int out_h = (image_size[1]+2*1 - kernel_size[2])/stride[0] + 1;
    int flat_kernel_size = kernel_size[3]*kernel_size[2]*kernel_size[1];
    int flat_img_size = out_w*out_h;
    flat_conv(Input, Kernel + flat_kernel_size*blockIdx.z , Output + flat_img_size*blockIdx.z, out_w);
}

__host__ float* convolution_relu(float* Input, float* Kernel, int* h_image_size, int* h_kernel_size, int h_pad,int h_stride)
{
    int out_w = (h_image_size[2]+2*1 - h_kernel_size[3])/h_stride + 1;
    int out_h = (h_image_size[1]+2*1 - h_kernel_size[2])/h_stride + 1;
    float* Output;
    
    cudaMemcpyToSymbol(image_size,h_image_size,sizeof(int)*3);
    cudaMemcpyToSymbol(kernel_size,h_kernel_size,sizeof(int)*4);
    cudaMemcpyToSymbol(stride,&h_stride,sizeof(int));
    
    cudaMalloc((void***)&Output,out_w*out_h*h_kernel_size[0]*sizeof(float));
    
    int t = log(512/float(h_image_size[0]))/log(2);
    t = pow(2,(t/2));
    int thread_size_w = t/h_stride;
    int thread_size_h = t/h_stride;
    int block_size_w = h_image_size[2]/t;
    int block_size_h = h_image_size[1]/t;
    int flat_kernel_size = thread_size_w* thread_size_h* h_image_size[0]*sizeof(float);

    dim3 threads_c(thread_size_w, thread_size_h, h_kernel_size[1]);
	dim3 grid_c(block_size_w,block_size_h,h_kernel_size[0]);

    conv <<< grid_c,threads_c,flat_kernel_size>>>(Input,Kernel,Output);
    cudaFree(Input);
    //cudaFree(Kernel);
    return Output;
}



void randomInit(float* data, int size)
{
    
	for (int i = 0; i < size; ++i)
		data[i] = (rand() / (float)RAND_MAX)- 0.5;
}
void OneInit(float* data, int size)
{
    
	for (int i = 0; i < size; ++i)
		data[i] = 1;
}
__host__ float* cuda_kernel_maker(int wo,int wi, int wh, int ww)
{
    float *h_kernel;
    int h_kernel_len = sizeof(float)*wo*wi*wh*ww;
    h_kernel = (float*)malloc(h_kernel_len);

    OneInit(h_kernel,h_kernel_len/sizeof(float));

    float *ckernel;
    cudaMalloc((void***)&ckernel,h_kernel_len);
    cudaMemcpy(ckernel,h_kernel,h_kernel_len,cudaMemcpyHostToDevice);
    free(h_kernel);
    
    return ckernel;
}
__host__ int main(void)
{
    int h_kernel_size[4] ={1,1,3,3}; //O I H W;

    float* h_img;
    int h_image_size[3] = {3,32,32};
    int h_img_len = sizeof(float)*h_image_size[0]*h_image_size[1]*h_image_size[2];
    h_img = (float*)malloc(h_img_len);
    OneInit(h_img,h_img_len/sizeof(float));
    //read_img(h_img,h_img_len/sizeof(float));
    float *cimg;
    cudaMalloc((void***)&cimg,h_img_len);
    cudaMemcpy(cimg,h_img,h_img_len,cudaMemcpyHostToDevice);


    clock_t start = clock();
    

    float *coimg1;
    //float *ckernel1 = cuda_kernel_maker(4,3,3,3);
    float *ckernel1 = cuda_kernel_maker(4,3,3,3);
    h_kernel_size[0]=4;h_kernel_size[1]=3;
    coimg1 = convolution_relu(cimg,ckernel1,h_image_size,h_kernel_size,2,2);
    h_image_size[0] = 4;h_image_size[1] = 16;h_image_size[2] = 16;
    cudaFree(ckernel1);
    
    
    // float *coimg2;
    // float *ckernel2 = cuda_kernel_maker(8,4,3,3);
    // h_kernel_size[0]=8;h_kernel_size[1]=4;
    // coimg2 = convolution_relu(coimg1,ckernel2,h_image_size,h_kernel_size,1,1);
    // h_image_size[0] = 8;h_image_size[1] = 32;h_image_size[2] = 32;
    // cudaFree(ckernel2);
    
    
    // float *coimg3;
    // coimg3 = maxpooling(coimg2,h_image_size,2);
    // h_image_size[0] = 8;h_image_size[1] = 16;h_image_size[2] = 16;

    
    // float *coimg4;
    // float *ckernel4 = cuda_kernel_maker(16,8,3,3);
    // h_kernel_size[0]=16;h_kernel_size[1]=8;
    // coimg4 = convolution_relu(coimg3,ckernel4,h_image_size,h_kernel_size,1,1);
    // h_image_size[0] = 16;h_image_size[1] = 16;h_image_size[2] = 16;
    // cudaFree(ckernel4); 

    // float *coimg5;
    // float *ckernel5 = cuda_kernel_maker(32,16,3,3);
    // h_kernel_size[0]=32;h_kernel_size[1]=16;
    // coimg5 = convolution_relu(coimg4,ckernel5,h_image_size,h_kernel_size,1,1);
    // h_image_size[0] = 32;h_image_size[1] = 16;h_image_size[2] = 16;
    // cudaFree(ckernel5); 

    // float *coimg6;
    // coimg6 = maxpooling(coimg5,h_image_size,2);
    // h_image_size[0] = 32;h_image_size[1] = 8;h_image_size[2] = 8;

    // float *coimg7;
    // float *ckernel7 = cuda_kernel_maker(32,32,3,3);
    // h_kernel_size[0]=32;h_kernel_size[1]=32;
    // coimg7 = convolution_relu(coimg6,ckernel7,h_image_size,h_kernel_size,1,2);
    // h_image_size[0] = 32;h_image_size[1] = 4;h_image_size[2] = 4;
    // cudaFree(ckernel7);


    // float *coimg8;
    // float *ckernel8 = cuda_kernel_maker(32,4,4,10);
    // int fc_kernel_size[2] = {10,32*4*4};
    // coimg8 = fully_connected(coimg7,ckernel8,fc_kernel_size);
    
    clock_t end = clock();

    float* h_out;
    int h_out_len = sizeof(float)*h_image_size[0]*h_image_size[1]*h_image_size[2];
    //int h_out_len = 10*sizeof(float);
    h_out = (float*)malloc(h_out_len);
    cudaMemcpy(h_out,coimg1,h_out_len,cudaMemcpyDeviceToHost);

    // for(int i =0;i<10;i ++)
    //     printf("%f ",h_out[i]);
    
    //FILE* fs;
    //fs = fopen("out_text.txt","w");
    int cnt = 0;
    for(int i = 0;i < h_image_size[0]; i++)
    {
        for(int j =0; j < h_image_size[1];j ++)
        {
            for(int k =0; k < h_image_size[2];k ++)
            {
                printf("%.1f ",h_out[cnt]);
                //fprintf(fs,"%f ",h_out[cnt]);
                cnt +=1;
            }
            printf("\n");
            //fprintf(fs,"\n");   
        }
        printf("\n");
        //fprintf(fs,"\n"); 
    }

    //fclose(fs);
    //cudaFree(cimg);
    //cudaFree(coimg7);
    //cudaFree(cimg_size);
    //cudaFree(ckernel_size);
    //cudaFree(cpad);
    //cudaFree(cstride);
    
    
    printf("\n%f",(float)(end - start)/CLOCKS_PER_SEC);
}