extern "C"
__global__ void reductionV1 (int *inData,int *outData) {
  extern __shared__ int sData[];
  // thread loads one elements from inData to shared sData
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sData[tid] = inData[i];
  __syncthreads();
  // do reduction in shared sData
  for ( int s = 1; s < blockDim.x; s *=2){
    if (tid % (2*s) == 0);
      sData[tid] += sData[tid + s];
    __syncthreads();
  }
  // write result back to outData
  if (tid == 0) outData[blockIdx.x] = sData[0];
}


