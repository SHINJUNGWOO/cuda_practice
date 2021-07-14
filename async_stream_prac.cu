#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__constant__  int image_size[3];
__constant__  int kernel_size[4];
__constant__  int stride[1];
__host__ float* cuda_kernel_maker(int wo,int wi, int wh, int ww,cudaStream_t stream);
typedef struct{
    float * kernel;
    float * output;
} return_out;

__device__ void flat_conv(float* Input, float* Kernel, float* Output,int* out_w)
{
    extern __shared__ float kernel_part[];

    int out_col_idx = (blockIdx.x*blockDim.x + threadIdx.x);
    int out_row_idx = (blockIdx.y*blockDim.y + threadIdx.y);

    int col_idx = stride[0]*out_col_idx;
    int row_idx = stride[0]*out_row_idx;

    kernel_part[threadIdx.y*blockDim.x + threadIdx.x + blockDim.x*blockDim.y*threadIdx.z] = 0;
    for(int i =-1; i < 2 ; i ++)
    {
        for(int j = -1; j<2 ; j++)
        {
            if ((col_idx+i) >= 0 && (row_idx+j) >= 0 && (col_idx+i) < image_size[2] && (row_idx+j) < image_size[1])
            {
                kernel_part[threadIdx.y*blockDim.x + threadIdx.x + blockDim.x*blockDim.y*threadIdx.z] +=
                Input[ (col_idx+i) * image_size[2] + row_idx+j + image_size[2]*image_size[1]*threadIdx.z] *
                Kernel[ (i+1)*kernel_size[3] +(j+1) + kernel_size[3]*kernel_size[2]*threadIdx.z];
            }
            else{
                kernel_part[threadIdx.y*blockDim.x + threadIdx.x + blockDim.x*blockDim.y*threadIdx.z] += 0;
            }

        }
    }
    __syncthreads();


    atomicAdd(&(Output[out_col_idx* out_w[0] +out_row_idx]),kernel_part[threadIdx.y*blockDim.x + threadIdx.x + blockDim.x*blockDim.y*threadIdx.z]);


}

__global__ void conv(float* Input, float* Kernel, float* Output)
{   
    int out_w = (image_size[2]+2 - kernel_size[3])/stride[0] + 1;
    int out_h = (image_size[1]+2 - kernel_size[2])/stride[0] + 1;
    int flat_kernel_size = 3*3*kernel_size[1];
    int flat_img_size = out_w*out_h;
    flat_conv(Input, Kernel + flat_kernel_size*blockIdx.z , Output + flat_img_size*blockIdx.z, &out_w);
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

__host__ return_out convolution_relu(float* Input, float* Kernel, int* h_image_size, int* h_kernel_size,int* h_out_kernel_size,int h_stride,cudaStream_t stream1,cudaStream_t stream2)
{
    int out_w =h_image_size[2]/h_stride; //(h_image_size[2]+2*1 - h_kernel_size[3])/h_stride + 1;
    int out_h =h_image_size[1]/h_stride; //(h_image_size[1]+2*1 - h_kernel_size[2])/h_stride + 1;
    float* Output;
    
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

    conv <<< grid_c,threads_c,flat_kernel_size,stream1>>>(Input,Kernel,Output);


    // int relu_t_wsize = h_image_size[2] > 32 ? 32: out_w;
    // int relu_t_hsize = h_image_size[1] > 32 ? 32: out_h;
    // dim3 threads_r(relu_t_wsize,relu_t_hsize);
	// dim3 grid_r(out_w/relu_t_wsize,out_h/relu_t_hsize,h_kernel_size[0]);
    // relu <<<grid_r,threads_r >>>(Output);
    h_image_size[0] = h_kernel_size[0];
    h_image_size[1] = out_w;
    h_image_size[2] = out_h;

    float* new_kernel = cuda_kernel_maker(h_out_kernel_size[0],h_out_kernel_size[1],h_out_kernel_size[2],h_out_kernel_size[3],stream2);
    cudaMemcpyToSymbolAsync(image_size,h_image_size,sizeof(int)*3,0,cudaMemcpyHostToDevice,stream2);
    cudaMemcpyToSymbolAsync(kernel_size,h_kernel_size,sizeof(int)*4,0,cudaMemcpyHostToDevice ,stream2);
    cudaMemcpyToSymbolAsync(stride,&h_stride,sizeof(int),0,cudaMemcpyHostToDevice,stream2);

    cudaFree(Input);
    cudaFree(Kernel);
    return_out out_val;
    out_val.output = Output;
    out_val.kernel = new_kernel;
    return out_val;
}

__global__ void g_maxpooling(float* Input,float* Output)
{
    
    int row_idx = blockIdx.y*blockDim.y +threadIdx.y;
    int col_idx = blockIdx.x*blockDim.x +threadIdx.x;
    int col_len = gridDim.x*blockDim.x;
    int row_len = gridDim.y*blockDim.x;
    int flat_img_size = col_len*row_len;
    extern __shared__ float kernel_part[];
    //size of double of kernel

    int shm_idx = blockDim.x*threadIdx.y + threadIdx.x +blockDim.x*blockDim.y*threadIdx.z;

    kernel_part[shm_idx] = Input[row_idx*col_len + col_idx + threadIdx.z*flat_img_size];
    __syncthreads();
    for (int size = blockDim.x/2; size>0; size=size/2) { 
        if (threadIdx.x < size)
        {
            kernel_part[shm_idx] =  kernel_part[shm_idx] >
                                    kernel_part[shm_idx + size] ?
                                    kernel_part[shm_idx] :
                                    kernel_part[shm_idx + size] ;
           

        } 
        __syncthreads();
    }
    for (int size = blockDim.y/2; size>0; size=size/2) { 
        if (threadIdx.y < size)
        {
            kernel_part[shm_idx] =  kernel_part[shm_idx] >
                                    kernel_part[shm_idx + size*blockDim.x] ?
                                    kernel_part[shm_idx] :
                                    kernel_part[shm_idx + size*blockDim.x] ;
            
        }
        __syncthreads();

    }
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        Output[blockIdx.y*gridDim.x+blockIdx.x + threadIdx.z*gridDim.x*gridDim.y] = kernel_part[blockDim.x*blockDim.y*threadIdx.z];
    }
}
__host__ float* maxpooling(float* Input, int* h_image_size, int h_kernel_size)
{
    float* Output;
    int col_len = h_image_size[2]/h_kernel_size;
    int row_len = h_image_size[1]/h_kernel_size;
    cudaMalloc((void***)&Output,row_len*col_len*h_image_size[0]*sizeof(float));
    dim3 threads(h_kernel_size,h_kernel_size,h_image_size[0]);
	dim3 grid(col_len,row_len);
    int shm_len = h_image_size[0]*h_kernel_size*h_kernel_size*sizeof(float);

    g_maxpooling<<<grid,threads,shm_len>>>(Input,Output);

    cudaFree(Input);
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
		data[i] = 0.07;
}
__host__ float* cuda_kernel_maker(int wo,int wi, int wh, int ww,cudaStream_t stream)
{
    float *h_kernel;
    int h_kernel_len = sizeof(float)*wo*wi*wh*ww;
    h_kernel = (float*)malloc(h_kernel_len);

    OneInit(h_kernel,h_kernel_len/sizeof(float));

    float *ckernel;
    cudaMalloc((void**)&ckernel,h_kernel_len);
    cudaMemcpyAsync(ckernel,h_kernel,h_kernel_len,cudaMemcpyHostToDevice,stream);
    free(h_kernel);

    return ckernel;
}
__host__ int main(void)
{
    int h_kernel_size[4] ={4,3,3,3}; //O I H W;
    int h_out_kernel_size[4] ={4,4,3,3};
    int h_stride =1;
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

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    
    float* kernel_1 = cuda_kernel_maker(h_out_kernel_size[0],h_out_kernel_size[1],h_out_kernel_size[2],h_out_kernel_size[3],stream1);
    cudaMemcpyToSymbolAsync(image_size,h_image_size,sizeof(int)*3,0,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyToSymbolAsync(kernel_size,h_kernel_size,sizeof(int)*4,0,cudaMemcpyHostToDevice ,stream1);
    cudaMemcpyToSymbolAsync(stride,&h_stride,sizeof(int),0,cudaMemcpyHostToDevice,stream1);

    return_out coimg1;
    //float *ckernel1 = cuda_kernel_maker(4,3,3,3);
    h_kernel_size[0]=4;h_kernel_size[1]=3;
    h_out_kernel_size[0]=8;h_out_kernel_size[1]=4;
    coimg1 = convolution_relu(cimg,kernel_1,h_image_size,h_kernel_size,h_out_kernel_size,1,stream1,stream2);
    h_image_size[0] = 4;h_image_size[1] = 32;h_image_size[2] = 32;

    
    return_out coimg2;
    h_kernel_size[0]=8;h_kernel_size[1]=4;
    h_out_kernel_size[0]=16;h_out_kernel_size[1]=8;
    coimg2 = convolution_relu(coimg1.output,coimg1.kernel,h_image_size,h_kernel_size,h_out_kernel_size,1,stream1,stream2);
    h_image_size[0] = 8;h_image_size[1] = 32;h_image_size[2] = 32;
    
    
    float *coimg3;
    coimg3 = maxpooling(coimg2.output,h_image_size,2);
    h_image_size[0] = 8;h_image_size[1] = 16;h_image_size[2] = 16;

    return_out coimg4;
    h_kernel_size[0]=16;h_kernel_size[1]=8;
    h_out_kernel_size[0]=32;h_out_kernel_size[1]=16;
    coimg4 =  convolution_relu(coimg3,coimg2.kernel,h_image_size,h_kernel_size,h_out_kernel_size,1,stream1,stream2);
    h_image_size[0] = 16;h_image_size[1] = 16;h_image_size[2] = 16;




    return_out coimg5;
    h_kernel_size[0]=32;h_kernel_size[1]=16;
    h_out_kernel_size[0]=32;h_out_kernel_size[1]=32;
    coimg5 = convolution_relu(coimg4.output,coimg4.kernel,h_image_size,h_kernel_size,h_out_kernel_size,1,stream1,stream2);
    h_image_size[0] = 32;h_image_size[1] = 16;h_image_size[2] = 16;

    float *coimg6;
    coimg6 = maxpooling(coimg5.output,h_image_size,2);
    h_image_size[0] = 32;h_image_size[1] = 8;h_image_size[2] = 8;

    return_out coimg7;
    h_kernel_size[0]=32;h_kernel_size[1]=32;
    h_out_kernel_size[0]=32;h_out_kernel_size[1]=32;
    coimg7 = convolution_relu(coimg6,coimg5.kernel,h_image_size,h_kernel_size,h_out_kernel_size,2,stream1,stream2);
    h_image_size[0] = 32;h_image_size[1] = 4;h_image_size[2] = 4;

;
    
    clock_t end = clock();

    float* h_out;
    int h_out_len = sizeof(float)*h_image_size[0]*h_image_size[1]*h_image_size[2];
    //int h_out_len = 10*sizeof(float);
    h_out = (float*)malloc(h_out_len);
    cudaMemcpy(h_out,coimg7.output,h_out_len,cudaMemcpyDeviceToHost);

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