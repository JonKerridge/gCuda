import gCuda.Dim3
import jcuda.driver.CUstream

//@gCudaDriverKernel
    Dim3 blockDim = new Dim3()
    Dim3 blockIdx = new Dim3()
    Dim3 threadIdx = new Dim3()
    int sharedMemoryBytes = 0
    CUstream hStream = null
    List extra = null

//@gCudaDataIn
    List<Float> a    // matrices stored linearly in ROW order
    List<Float> b    // only one property per line
    int mSize

//@gCudaDataOut
    List<Float> c

//@gCudaDataInOut

// add the parameter variables, comma separated,
//in the order they appear in the kernel definition
//to the end of the following line
//@gCudaKernelParams a, b, c, mSize

//@gCudaKernel<
    def matrixMultiply = {List<Float> A, List<Float> B, List<Float> C, int N ->
      int row = blockIdx.y * blockDim.y + threadIdx.y
      int col = blockIdx.x * blockDim.x + threadIdx.x
      if (row < N && col < N) {
        float sum = 0.0f
        for (int i = 0; i < N; ++i) {
          sum += A[row * N + i] * B[i * N + col]
        }
        C[row * N + col] = sum
      }
    } //matrixMultiply

//@>

// initialise the host data
    a = [1.0f, 2.0f, 3.0f, 4.0f]
    b = [5.0f, 6.0f, 7.0f, 8.0f]
    c = new float[4]
    mSize = 2

    blockDim.x = 16
    blockDim.y = 16
    blockIdx.x = (int) ((mSize + blockDim.x - 1) / blockDim.x)
    blockIdx.y = (int) ((mSize + blockDim.y - 1) / blockDim.y)

    println ("sizes: blockDim = " + blockDim + ", blockIdx = " +blockIdx)

//@gCudaLaunchKernel matrixMultiply

// if required emulate the whole process
    List<Float> localC
    localC = new Float[4]
    for (bIdy in 0 ..< blockIdx.y) {
      blockIdx.y = bIdy
      for (bIdx in 0 ..< blockIdx.x) {
        blockIdx.x = bIdx
        for (tIdx in 0 ..< blockDim.x) {
          threadIdx.x = tIdx
          for (tIdy in 0 ..<  blockDim.y){
            threadIdx.y = tIdy
            matrixMultiply(a, b, localC, mSize)
          }
        }
      }
    }

//@gCudaEndKernel

// now add any final host coding
    println "finished"
    for ( i in 0 ..< (mSize * mSize)) print "${localC[i]}, "
    println ""
    for ( i in 0 ..< (mSize * mSize)) print "${c[i]}, "
    println ""