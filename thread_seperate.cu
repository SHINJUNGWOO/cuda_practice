#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 32
#define WA 64   
#define HA 64     
#define HC 3     
#define WC 3
#define PAD 1
#define WB (WA+2*PAD - WC + 1)
#define HB (HA+2*PAD - HC + 1)
#define CHANNEL_SIZE 3
__global__ void Convolution(float* Input, float* Kernel, float* Output)
{
    __shared__ float kernel_part[HC][WC][CHANNEL_SIZE];
    int col_idx = blockIdx.x - PAD + threadIdx.x;
    int row_idx = blockIdx.y - PAD + threadIdx.y;
    if( WA>col_idx && col_idx >=0 && HA>row_idx && row_idx >=0)
    {
        kernel_part[threadIdx.y][threadIdx.x][threadIdx.z] = Input[(col_idx * WA +row_idx)*CHANNEL_SIZE + threadIdx.z];
    }
    else
    {
        kernel_part[threadIdx.y][threadIdx.x][threadIdx.z] = 0;
    }
    //__syncthreads;

    atomicAdd(&(Output[(blockIdx.x * WB +blockIdx.y)]), kernel_part[threadIdx.y][threadIdx.x][threadIdx.z]);
}

__host__ int main(void)
{


    


    float h_a[64][64][3] ={1.0,1.0,1.0};
    float h_b[3] [3] [3] ={ 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};

    float h_c[64][64] ={0.0};

    float *da;
    float *db;
    float *dc;
    cudaMalloc((void***)&da,sizeof(h_a));
    cudaMalloc((void***)&db,sizeof(h_b));
    cudaMalloc((void***)&dc,sizeof(h_c));

    cudaMemcpy(da,h_a,sizeof(h_a),cudaMemcpyHostToDevice);
    cudaMemcpy(db,h_b,sizeof(h_b),cudaMemcpyHostToDevice);

    dim3 threads(WC, HC,CHANNEL_SIZE);
	dim3 grid(WB,HB);
    Convolution <<< grid,threads>>>(da,db,dc);
    
    cudaMemcpy(h_c,dc,sizeof(h_c),cudaMemcpyDeviceToHost);

    for(int j =0; j < WB;j ++)
    {
        for(int k =0; k < WB;k ++)
        {
            printf("%.0f ",h_c[k][j]);
        }
        printf("\n");   
    }
    printf("\n");


}