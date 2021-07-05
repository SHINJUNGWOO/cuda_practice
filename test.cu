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


__global__ void conv_flat(float* A, float* B, float* C)
{
	int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
	int row_i = row - WC + 1;
	int col_i = col - WC + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE][CHANNEL_SIZE];

	if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0)
	{   
        shm[threadIdx.y][threadIdx.x][blockIdx.z] = A[(col_i * WA + row_i)*CHANNEL_SIZE +blockIdx.z];
		
	}
	else
    {        

        shm[threadIdx.y][threadIdx.x][blockIdx.z] = 0;
            
	}
	__syncthreads();

    if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) && col < (WB - WC + 1))
	{
		for (int i = 0; i< WC;i++)
        {
			for (int j = 0;j<WC;j++)
            {
                    tmp += shm[threadIdx.y + i][threadIdx.x + j][blockIdx.z] * C[(j*WC + i)*CHANNEL_SIZE+blockIdx.z];
            }
        }
		B[(col*WB + row)*CHANNEL_SIZE + blockIdx.z] = tmp;
	}


}
__global__ void hadamad(float* channel_expanded,float* out_val)
{
    
    out_val[blockIdx.x * WB +blockIdx.y] += channel_expanded[(blockIdx.x * WB +blockIdx.y) *CHANNEL_SIZE + threadIdx.x];
    __syncthreads();
}

void Convolution(float* A, float* B, float* C)
{
    float *channel_expanded;
    cudaMalloc( &channel_expanded, WB*HB*CHANNEL_SIZE*sizeof(float) );

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid( (WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1),CHANNEL_SIZE);
    conv_flat <<<grid,threads>>>(A, channel_expanded,C);

    hadamad <<<grid,CHANNEL_SIZE>>>(channel_expanded, B);

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
    // dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	// dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1),CHANNEL_SIZE);
    Convolution(da,dc,db);
    
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