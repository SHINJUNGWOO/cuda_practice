# cuda_practice

version 1:

practice code :
thread_seperate.cu
test.cu
conv_with_out_channel

Done:
conv_with_stride.cu
model.cuh
model_sequence.cu


nvcc model_sequence.cu conv_with_stride.cu

version2:
new_conv.cu


version3

async_stream_prac.cu

version 1의 Thread 나누는 것 및 연산의 메모리 접근이 비효율적
version 2에서 Thread 나누는 방식의 변경 및 로컬 메모리의 사용
version 3에서 메모리 입력 및 kernel 생성 Stream 병렬화 처리
