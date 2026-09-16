extern "C"
__global__ void scalerMultiply (int s,int vSize,float *vector) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < vSize){
    vector[i] = vector[i] * s;
  }
} // scalerMultiply

