import gCuda.Dim3
import jcuda.Pointer
import jcuda.driver.*

//@gcDriverKernel   src/test/groovy/gCudaScripts/  StaticSharedMemory 61

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


//@gcKernelDefinition C
def staticReverse = {int[] d, int n ->
  //@gc:__shared__ int s[64];
  int t = threadIdx.x
  int tr = n-t-1
  s[t] = d[t]
  //@gc:__syncthreads();
  d[t] = s[tr]
}

//@gcDataToGPU
int n = 64  // only works for n = 32 and 64 on a GT1030

//@gcDataFromGPU

//@gcDataBoth
int[] d = new int[n]

//@gcDataInitialise
int[] r = new int[n]
for (i in 0 ..< n){
  d[i] = i
  r[i] = n-i-1
}

blockSize.x = n
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) n / blockSize.x)
println "Blocksize = ${blockSize.x}, gridsize = ${gridSize.x}"

gpuStart = System.currentTimeMillis()

//@gcKernelParams d,n

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()

//@gcEmulate

blockDim = blockSize

emulateEnd = System.currentTimeMillis()

//@gcFinalise
for ( i in 0 ..< n){
  assert (d[i] == r[i]):
      "At index $i found ${d[i]} but expected ${r[i]}"
}

verifyEnd = System.currentTimeMillis()

//@gcFinish
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


