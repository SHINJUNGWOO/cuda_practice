#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 32
#define WA 64   
#define HA 64     
#define HC 3     
#define WC 3
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)
#define CHANNEL_SIZE 3

__global__ void Convolution(float* A, float* B, float* C)
{
	int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
	int row_i = row - WC + 1;
	int col_i = col - WC + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE][CHANNEL_SIZE];

	if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0)
	{   
        for (int i =0 ; i < CHANNEL_SIZE; i ++)
        {
            shm[threadIdx.y][threadIdx.x][i] = A[(col_i * WA + row_i)*CHANNEL_SIZE +i];
        }
		
	}
	else
    {        
        for (int i =0 ; i < CHANNEL_SIZE; i ++)
        {
            shm[threadIdx.y][threadIdx.x][i] = 0;
        }
            
	}
	__syncthreads();

	if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) && col < (WB - WC + 1))
	{
		for (int i = 0; i< WC;i++)
			for (int j = 0;j<WC;j++)
                for(int k =0; k<CHANNEL_SIZE; k++)
                {
                    tmp += shm[threadIdx.y + i][threadIdx.x + j][k] * C[(j*WC + i)*CHANNEL_SIZE+k];
                }
		B[col*WB + row] = tmp;
	}
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}
__host__ int main(void)
{
    float h_a[64][64][3] ={1.0,1.0,1.0};
    float h_b[3] [3] [3] ={ 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};

    float h_c[62][62] ={0.0};

    float *da;
    float *db;
    float *dc;
    cudaMalloc((void***)&da,sizeof(h_a));
    cudaMalloc((void***)&db,sizeof(h_b));
    cudaMalloc((void***)&dc,sizeof(h_c));

    cudaMemcpy(da,h_a,sizeof(h_a),cudaMemcpyHostToDevice);
    cudaMemcpy(db,h_b,sizeof(h_b),cudaMemcpyHostToDevice);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1));
    Convolution <<< grid,threads>>>(da,dc,db);
    
    cudaMemcpy(h_c,dc,sizeof(h_c),cudaMemcpyDeviceToHost);

    for(int j =0; j < 62;j ++)
    {
        for(int k =0; k < 62;k ++)
        {
            printf("%.0f ",h_c[k][j]);
        }
        printf("\n");   
    }
    printf("\n");


}
