extern "C"
__global__ void scalerMultiply (int s,int vSize,float *vector) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < vSize){
    vector[i] = (float) (vector[i] * s);
  }
} // scalerMultiply

