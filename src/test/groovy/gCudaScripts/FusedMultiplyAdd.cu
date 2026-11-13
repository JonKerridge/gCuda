extern "C"
__global__ void fusedMultiplyAdd (float *a,float *b,float *c,float *d,int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float sum = 0.0f;
    for ( int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * n + col];
    }
    d[row * n + col] = sum + c[row * n + col];
  }
} // fusedMultiplyAdd

