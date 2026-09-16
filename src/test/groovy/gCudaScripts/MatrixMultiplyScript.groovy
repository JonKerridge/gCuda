import gCuda.Dim3
import jcuda.Pointer
import jcuda.driver.*

import javax.swing.MenuSelectionManager

//@gcDriverKernel   src/test/groovy/gCudaScripts  MatrixMultiply 61

Dim3 gridSize = new Dim3()    // must be initialised in the DataInitialise phase
Dim3 blockSize = new Dim3()   //must be a multiple of 32
int sharedMemoryBytes = 0
CUstream hStream = null
Pointer extra = null

// CUDA idiomatic properties
Dim3 blockIdx = new Dim3()
Dim3 blockDim = new Dim3()
Dim3 threadIdx = new Dim3()
long startTime, gpuStart, gpuEnd, emulateEnd, verifyEnd
startTime = System.currentTimeMillis()

//@gcKernelDefinition
def matrixMultiply = {float[] A, float[] B, float[] C, int N ->
  int row = blockIdx.y * blockDim.y + threadIdx.y
  int col = blockIdx.x * blockDim.x + threadIdx.x
  if (row < N && col < N) {
    float sum = 0.0f
    for ( int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * N + col]
    }
    C[row * N + col] = sum
  }
} //matrixMultiply

//@gcDataToGPU
int mSize = 2   // needs to be much larger to make use of a GPU
float[] a = new float[mSize * mSize]  // matrices stored linearly in ROW order
float[] b = new float[mSize * mSize]      // only one property per line



//@gcDataFromGPU
float[] c = new float[mSize * mSize]

//@gcDataBoth

//@gcDataInitialise

for ( i in 0 ..< mSize * mSize){
  a[i] = (float) i
  b[i] = (float) i
  c[i] = 0.0f
}

blockSize.x = 16   // must be a multiple of 32, depends on GPU used
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) mSize / blockSize.x)
blockSize.y = 16
gridSize.y = (int)Math.ceil((double) mSize / blockSize.y)
println " blockSize = $blockSize, gridSize = $gridSize, mSize = $mSize"
gpuStart = System.currentTimeMillis()

//@gcKernelParams a,b,c,mSize

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()
//@gcEmulate
blockDim.x = blockSize.x
blockDim.y = blockSize.y
blockDim.z = blockSize.z

float[] lc = new float[mSize*mSize]
for ( gy in 0 ..< gridSize.y) {
  blockIdx.y = gy
  for (gx in 0 ..< gridSize.x) {
    blockIdx.x = gx
    for (ty in 0 ..< blockSize.y){
      threadIdx.y = ty
      for ( tx in 0 ..< blockSize.x){
        threadIdx.x = tx
        matrixMultiply(a, b, lc, mSize)
      }
    }
  }
}
emulateEnd = System.currentTimeMillis()
//@gcFinalise

//c = [2, 3, 6, 11]   //for local testing remove when using GPU

for ( i in 0 ..< mSize* mSize){
  assert Math.abs(lc[i] - c[i]) < 1e-5 :
      "At index $i found ${lc[i]} but expected ${c[i]}"
}

verifyEnd = System.currentTimeMillis()
//@gcFinish
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"

