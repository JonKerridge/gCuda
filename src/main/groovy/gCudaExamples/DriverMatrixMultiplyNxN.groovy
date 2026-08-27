package gCudaExamples

import gCuda.Dim3

//class DriverMatrixMultiplyNxN {
//  static void main(String[] args) {
// kernel and driver version of Matric Multiply
// C = A x B square mSize matrices

//@gCudaDriverKernel
    Dim3 blockDim = new Dim3()
    Dim3 blockIdx = new Dim3()
    Dim3 threadIdx = new Dim3()
    int sharedMemoryBytes = 0
    Long hStream = 0  //  a CUstream of commands from host to device default null
    List extra = null     // any axtra required parameters


//@gCudaDataIn
    List<Float> a,b    // matrices stored linearly in ROW order
    int mSize

//@gCudaDataOut
    List<Float> c

//@gCudaDataInOut

//@gCudaKernelParams a, b, c, mSize

//@gCudaKernel<
  def matrixMultiply = {List<Float> A, List<Float> B, List<Float> C, int N ->
      int row = blockIdx.y * blockDim.y + threadIdx.y
      int col = blockIdx.x * blockDim.x + threadIdx.x
//  print "[$row, $col], "

      if (row < N && col < N) {
        float sum = 0.0f
        for (int i = 0; i < N; ++i) {
//      print "${A[row * N + i]} * ${B[i * N + col]}, $row, $col, $i, "
          sum += A[row * N + i] * B[i * N + col]
        }
        C[row * N + col] = sum
//    println "${row * N + col}, $sum"
      }
  } //matrixMultiply

//@>

// initialise the host data in ROW order
    a = [1.0f, 2.0f, 3.0f, 4.0f]
    b = [5.0f, 6.0f, 7.0f, 8.0f]
    c = new float[4]
    mSize = 2


    blockDim.x = 16
    blockDim.y = 16
    blockIdx.x = (int) ((mSize + blockDim.x - 1) / blockDim.x)
    blockIdx.y = (int) ((mSize + blockDim.y - 1) / blockDim.y)

    println "sizes: blockDim =${blockDim}, blockIdx = ${blockIdx}"


//@gCudaLaunchKernel matrixMultiply


// if required emulate the whole process


    for (bIdy in 0 ..< blockIdx.y) {
      blockIdx.y = bIdy
      for (bIdx in 0 ..< blockIdx.x) {
        blockIdx.x = bIdx
        for (tIdx in 0 ..< blockDim.x) {
          threadIdx.x = tIdx
          for (tIdy in 0 ..<  blockDim.y){
            threadIdx.y = tIdy
            matrixMultiply(a, b, c, mSize)
          }
        }
      }
    }

//@gCudaEndKernel

// now add any final host coding

    println "Finished"
    for ( i in 0 ..< (mSize * mSize)) print "${c[i]}, "
    println ""

//  } //main
//} //class
