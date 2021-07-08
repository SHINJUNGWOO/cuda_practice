

#include "model.cuh"


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

    int relu_t_wsize = h_image_size[2] > 32 ? 32: out_w;
    int relu_t_hsize = h_image_size[1] > 32 ? 32: out_h;
    dim3 threads_r(relu_t_wsize,relu_t_hsize);
	dim3 grid_r(out_w/relu_t_wsize,out_h/relu_t_hsize,h_kernel_size[0]);
    relu <<<grid_r,threads_r >>>(Output);
    
    cudaFree(Input);
    //cudaFree(Kernel);
    return Output;
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


__global__ void fc(float* Input,float* Kernel, float* Output)
{
    extern __shared__ float kernel_part[];
    kernel_part[threadIdx.x] = Input[threadIdx.x] * Kernel[blockDim.x*blockIdx.x+threadIdx.x];

    atomicAdd(&Output[blockIdx.x],kernel_part[threadIdx.x]);
}

__host__ float* fully_connected(float* Input,float* Kernel,int* kernel_size)
{   // Weight size = (Out channel, In channel)
    
    float* Output;
    cudaMalloc((void**)&Output,kernel_size[0]*sizeof(float));
    int shm_len = kernel_size[1]*sizeof(float);
    dim3 thread(kernel_size[1]);
    dim3 grid(kernel_size[0]);
    fc<<<grid,thread,shm_len>>>(Input,Kernel,Output);
    return Output;
}




/*
__host__ int main()
{
    float* h_img; float *h_kernel; float* h_out;
    int h_img_len = sizeof(float)*10;
    int h_kernel_len = sizeof(float)*10*10;
    int h_out_len = sizeof(float)*10;

    h_img = (float*)malloc(h_img_len);
    h_kernel = (float*)malloc(h_kernel_len);
    h_out = (float*)malloc(h_out_len);
    OneInit(h_kernel,h_kernel_len/sizeof(float));
    OneInit(h_img,h_img_len/sizeof(float));
    h_kernel[0] = 10;
    float *cimg;
    float *ckernel;
    cudaMalloc((void**)&cimg,h_img_len);
    cudaMalloc((void**)&ckernel,h_kernel_len);
    cudaMemcpy(cimg,h_img,h_img_len,cudaMemcpyHostToDevice);
    cudaMemcpy(ckernel,h_kernel,h_kernel_len,cudaMemcpyHostToDevice);
    int w_s[2] = {10,10};
    float *cout;
    cout = fully_connected(cimg,ckernel,w_s);
    cudaMemcpy(h_out,cout,h_out_len,cudaMemcpyDeviceToHost);
    for(int i =0; i <10; i++)
    {
        printf("%.0f ",h_out[i]);
    }


}
*/