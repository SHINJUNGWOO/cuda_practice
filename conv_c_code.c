#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define C1_N_CHAN 3
#define C1_X_DMNIN 32
#define C1_N_FILTERS 3
#define C1_W_DMNIN 4
#define C1_OUT_DMNIN 30
#define STRIDE 1

#pragma unroll


int main()
{


    float   X[C1_N_CHAN][C1_X_DMNIN][C1_X_DMNIN];
    float   W[C1_N_CHAN][C1_N_FILTERS][C1_W_DMNIN][C1_W_DMNIN];
    float out[C1_N_FILTERS][C1_OUT_DMNIN][C1_OUT_DMNIN];

    

    uint8_t ch, f, i, j, r, c,k,h;


    for(i =0;i<C1_N_CHAN;i++)
        for(j=0;j<C1_X_DMNIN;j++)
            for(k=0;k<C1_X_DMNIN;k++)
            {
                X[i][j][k] = 1;
            }
    for(i =0;i<C1_N_CHAN;i++)
        for(j=0;j<C1_N_FILTERS;j++)
            for(k=0;k<C1_W_DMNIN;k++)
                for(h=0;h<C1_W_DMNIN;h++)
                {
                    W[i][j][k][h] = 1;
                }

    clock_t start = clock();
    for (f = 0; f < C1_N_FILTERS; ++f) {
        #pragma unroll
        for (r = 0; r < C1_OUT_DMNIN; ++r) {
            #pragma unroll
            for (c = 0; c < C1_OUT_DMNIN; ++c) {
                out[f][r][c] = 0;
            }
        }

        for (ch = 0; ch < C1_N_CHAN; ++ch) {
            #pragma unroll
            for (r = 0; r < C1_X_DMNIN - C1_W_DMNIN + 1; r += STRIDE) {
                //for (c = 0, i = 0, j = 0; c < C1_X_DMNIN - C1_W_DMNIN + 1; c += STRIDE) {
                #pragma unroll
                for (c = 0; c < C1_X_DMNIN - C1_W_DMNIN + 1; c += STRIDE) {
                    #pragma unroll
                    for (i = 0; i < C1_W_DMNIN; ++i) {
                        #pragma unroll
                        for (j = 0; j < C1_W_DMNIN; ++j) {
                            out[f][r][c] += X[ch][r + i][j + c] * W[ch][f][i][j];
                        }
                    }
                }
            }
        }
    }

    clock_t end = clock();

    for(i =0;i<C1_N_FILTERS;i++)
    {
        for(j=0;j<C1_OUT_DMNIN;j++)
        {
            for(k=0;k<C1_OUT_DMNIN;k++)
            {
                printf("%f ",out[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n%f",(float)(end - start)/CLOCKS_PER_SEC);
    return 0;
}