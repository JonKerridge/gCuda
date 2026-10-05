extern "C"
__global__ void reductionV4 (int *inData,int *outData) {
  extern __shared__ int sData[];
  // thread loads one elements from inData to shared sData
  int tid = threadIdx.x;
  int i = (blockIdx.x * blockDim.x * 2 ) + threadIdx.x;
  sData[tid] = inData[i] + inData[i+blockDim.x];
  __syncthreads();
  // do reduction in shared sData
  for ( int s = blockDim.x / 2; s > 0; s >>= 1){
    if (tid < s);
      sData[tid] += sData[tid + s];
    __syncthreads();
  }
  // write result back to outData
  if (tid == 0) outData[blockIdx.x] = sData[0];
}


