package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.driver.*

import javax.swing.MenuSelectionManager

//@gcDriverKernel   src/test/groovy/gCudaScripts  AsymMatrixMultiply

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
def AsymMatrixMultiply = {float[] A, float[] B, float[] C, int M, int K, int N ->
  // C[r][c] = sumOverK of A[i][k] * B[k][j]
  int row = blockIdx.y * blockDim.y + threadIdx.y
  int col = blockIdx.x * blockDim.x + threadIdx.x
  if (row < M && col < N) {
    float sum = 0.0f
    for ( int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col]
    }
    C[row * N + col] = sum
  }
} //matrixMultiply

//@gcDataToGPU
int M = 200
int K  = 100    // AWidth = BHeight or A columns = B rows
int N  = 150
float[] a = new float[M * K]  // matrices stored linearly in ROW order
float[] b = new float[K * N]  // only one property per line



//@gcDataFromGPU
float[] c = new float[M * N]

//@gcDataBoth

//@gcDataInitialise

for ( i in 0 ..< M * K) a[i] = (float) i
for ( i in 0 ..< K * N) b[i] = (float) i
for ( i in 0 ..< M * N) c[i] = 0.0f


blockSize.x = 32
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) K*N / blockSize.x)
blockSize.y = 32
gridSize.y = (int)Math.ceil((double) M*K / blockSize.y)
println " blockSize = $blockSize, gridSize = $gridSize, M=$M, K=$K, N=$N"
gpuStart = System.currentTimeMillis()

//@gcKernelParams a, b, c, M, K ,N

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()
//@gcEmulate
blockDim.x = blockSize.x
blockDim.y = blockSize.y
blockDim.z = blockSize.z

float[] localOutput = new float[M*N]
for (row in 0 ..< M){
  for (col in 0 ..< N){
    float cVal = 0.0f
    for ( k in 0 ..< K){
      cVal = cVal + a[row * K + k] * b[k * N + col]
    }
    localOutput[row * N + col] = cVal
  }
}
emulateEnd = System.currentTimeMillis()
//@gcFinalise
println "finished"
boolean passed = true
for ( i in 0 ..< M*N)
  assert Math.abs(localOutput[i] - c[i]) < 1e-5 : "At index $i found ${c[i]} but expected ${localOutput[i]}"

verifyEnd = System.currentTimeMillis()
//@gcFinish
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


