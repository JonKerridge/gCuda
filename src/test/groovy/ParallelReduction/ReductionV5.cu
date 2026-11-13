__device__ void warpReduce ( volatile int *sdata, int tid) {
  sdata[tid] += sdata[tid+32];
  sdata[tid] += sdata[tid+16];
  sdata[tid] += sdata[tid+8];
  sdata[tid] += sdata[tid+4];
  sdata[tid] += sdata[tid+2];
  sdata[tid] += sdata[tid+1];
}
extern "C"
__global__ void reductionV5 ( int *inData, int *outData ) {
  extern __shared__ int sData[];
  // thread loads one elements from inData to shared sData
  int tid = threadIdx.x;
  int i = (blockIdx.x * blockDim.x * 2 ) + threadIdx.x;
  sData[tid] = inData[i] + inData[i+blockDim.x];
  __syncthreads();
  // do reduction in shared sData
  for ( int s = blockDim.x / 2; s > 32; s >>= 1){
    if (tid < s)
      sData[tid] += sData[tid + s];
    __syncthreads();
  }
  // write result back to outData
  if (tid < 32) warpReduce(sData,tid);
  if (tid == 0) outData[blockIdx.x] = sData[0];
}

