extern "C"
__global__ void AsymMatrixMultiply (float *A,float *B,float *C,int M,int K,int N) {
  // C[r][c] = sumOverK of A[i][k] * B[k][j]
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for ( int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
} //matrixMultiply

