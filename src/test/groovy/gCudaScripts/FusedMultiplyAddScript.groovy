import gCuda.Dim3
import jcuda.Pointer
import jcuda.driver.*

//@gcDriverKernel   src/test/groovy/gCudaScripts  FusedMltplyAdd 61

Dim3 gridSize = new Dim3()    // must be initialised in the DataInitialise phase
Dim3 blockSize = new Dim3()
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
def fusedMultiplyAdd = {float[] a, float[] b, float[] c, float[] d, int n ->
  int row = blockIdx.y * blockDim.y + threadIdx.y
  int col = blockIdx.x * blockDim.x + threadIdx.x
  if (row < n && col < n) {
    float sum = 0.0f
    for ( int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * n + col]
    }
    d[row * n + col] = sum + c[row * n + col]
  }
} // fusedMultiplyAdd

//@gcDataToGPU
int mSize = 2   // needs to be much larger to make use of a GPU
float[] a = new float[mSize * mSize]  // matrices stored linearly in ROW order
float[] b = new float[mSize * mSize]
float[] c = new float[mSize * mSize]

//@gcDataFromGPU
float[] d = new float[mSize * mSize]

//@gcDataBoth

//@gcDataInitialise
for ( i in 0 ..< mSize * mSize){
  a[i] = (float) i
  b[i] = (float) i
  c[i] = (float) i
  d[i] = 0.0f
}

blockSize.x = 16
gridSize.x = (int)Math.ceil((double) mSize / blockSize.x)
blockSize.y = 16
gridSize.y = (int)Math.ceil((double) mSize / blockSize.y)
println " blockSize = $blockSize, gridSize = $gridSize, mSize = $mSize"
gpuStart = System.currentTimeMillis()

//@gcKernelParams a,b,c,d,mSize

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()

//@gcEmulate
blockDim = blockSize

float[] ld = new float[mSize*mSize]
for ( gy in 0 ..< gridSize.y) {
  blockIdx.y = gy
  for (gx in 0 ..< gridSize.x) {
    blockIdx.x = gx
    for (ty in 0 ..< blockSize.y){
      threadIdx.y = ty
      for ( tx in 0 ..< blockSize.x){
        threadIdx.x = tx
        fusedMultiplyAdd(a, b, c, ld, mSize)
      }
    }
  }
}
emulateEnd = System.currentTimeMillis()

//@gcFinalise

d = [2, 4, 8, 14]   //for local testing comment out when using GPU
for ( i in 0 ..< mSize* mSize){
  assert Math.abs(ld[i] - d[i]) < 1e-5 :
      "At index $i found ${d[i]} but expected ${ld[i]}"
}
verifyEnd = System.currentTimeMillis()

//@gcFinish
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"
