#include "model.cuh"
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

#define OIMW 4
#define OIMH 4


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

    randomInit(h_kernel,h_kernel_len/sizeof(float));

    float *ckernel;
    cudaMalloc((void***)&ckernel,h_kernel_len);
    cudaMemcpy(ckernel,h_kernel,h_kernel_len,cudaMemcpyHostToDevice);
    free(h_kernel);
    
    return ckernel;
}

__host__ int main(void)
{
    int h_kernel_size[4] ={WO,WI,WH,WW}; //O I H W;
    int h_image_size[3] = {IMC,IMH,IMW}; //  O H W;

    float* h_img;
    int h_img_len = sizeof(float)*IMC*IMH*IMW;
    h_img = (float*)malloc(h_img_len);
    randomInit(h_img,h_img_len/sizeof(float));
    float *cimg;
    cudaMalloc((void***)&cimg,h_img_len);
    cudaMemcpy(cimg,h_img,h_img_len,cudaMemcpyHostToDevice);




    


    clock_t start = clock();
    //
    float *coimg1;
    float *ckernel1 = cuda_kernel_maker(4,3,3,3);
    h_kernel_size[0]=4;h_kernel_size[1]=3;
    coimg1 = convolution_relu(cimg,ckernel1,h_image_size,h_kernel_size,1,1);
    h_image_size[0] = 4;h_image_size[1] = 32;h_image_size[2] = 32;
    cudaFree(ckernel1);

    float *coimg2;
    float *ckernel2 = cuda_kernel_maker(8,4,3,3);
    h_kernel_size[0]=8;h_kernel_size[1]=4;
    coimg2 = convolution_relu(coimg1,ckernel2,h_image_size,h_kernel_size,1,1);
    h_image_size[0] = 8;h_image_size[1] = 32;h_image_size[2] = 32;
    cudaFree(ckernel2);
    
    
    float *coimg3;
    coimg3 = maxpooling(coimg2,h_image_size,2);
    h_image_size[0] = 8;h_image_size[1] = 16;h_image_size[2] = 16;

    
    float *coimg4;
    float *ckernel4 = cuda_kernel_maker(16,8,3,3);
    h_kernel_size[0]=16;h_kernel_size[1]=8;
    coimg4 = convolution_relu(coimg3,ckernel4,h_image_size,h_kernel_size,1,1);
    h_image_size[0] = 16;h_image_size[1] = 16;h_image_size[2] = 16;
    cudaFree(ckernel4); 

    float *coimg5;
    float *ckernel5 = cuda_kernel_maker(32,16,3,3);
    h_kernel_size[0]=32;h_kernel_size[1]=16;
    coimg5 = convolution_relu(coimg4,ckernel5,h_image_size,h_kernel_size,1,1);
    h_image_size[0] = 32;h_image_size[1] = 16;h_image_size[2] = 16;
    cudaFree(ckernel5); 

    float *coimg6;
    coimg6 = maxpooling(coimg5,h_image_size,2);
    h_image_size[0] = 32;h_image_size[1] = 8;h_image_size[2] = 8;

    float *coimg7;
    float *ckernel7 = cuda_kernel_maker(32,32,3,3);
    h_kernel_size[0]=32;h_kernel_size[1]=32;
    coimg7 = convolution_relu(coimg6,ckernel7,h_image_size,h_kernel_size,1,2);
    h_image_size[0] = 32;h_image_size[1] = 4;h_image_size[2] = 4;
    cudaFree(ckernel7);


    //float *coimg_6;
    //coimg_6 = fully_connected(coimg_6,ckernel,h_image_size,h_kernel_size,1,2);
    clock_t end = clock();

    float* h_out;
    int h_out_len = sizeof(float)*h_image_size[0]*h_image_size[1]*h_image_size[2];
    h_out = (float*)malloc(h_out_len);

    cudaMemcpy(h_out,coimg7,h_out_len,cudaMemcpyDeviceToHost);

    int cnt = 0;
    for(int i = 0;i < h_image_size[0]; i++)
    {
        for(int j =0; j < h_image_size[1];j ++)
        {
            for(int k =0; k < h_image_size[2];k ++)
            {
                //printf("%.0f ",h_c[cnt]);
                printf("%.1f ",h_out[cnt]);
                cnt +=1;
            }
            printf("\n");   
        }
        printf("\n");
    }

    //cudaFree(cimg);
    cudaFree(coimg7);
    //cudaFree(cimg_size);
    //cudaFree(ckernel_size);
    //cudaFree(cpad);
    //cudaFree(cstride);
    
    
    printf("%f",(float)(end - start)/CLOCKS_PER_SEC);
}