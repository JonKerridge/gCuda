import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

//@gcDriverKernel   src/test/groovy/gCudaScripts  ScalerMultiply

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
def scalerMultiply = {int s, int vSize, float[] vector ->
  int i = blockIdx.x * blockDim.x + threadIdx.x
  if ( i < vSize){
    vector[i] = vector[i] * s
  }
} // scalerMultiply

//@gcDataToGPU
int s = 2
int vSize = 200000000
//@gcDataFromGPU

//@gcDataBoth
float[] vector = new float[vSize]
//@gcDataInitialise
for ( i in 0 ..< vSize) vector[i] = (float)i

blockSize.x = 384
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) vSize / blockSize.x)
println " blockSize = $blockSize, gridSize = $gridSize, s=$s, vSize = $vSize"
gpuStart = System.currentTimeMillis()
//@gcKernelParams s, vSize, vector

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()
//@gcEmulate
float[] localV = new float[vSize]
for ( i in 0 ..< vSize) localV[i] = (float) i * s
emulateEnd = System.currentTimeMillis()
//@gcFinalise
for ( i in 0 ..<vSize)
  assert (localV[i] - vector[i]) < 1e-5 : "At index $i got ${vector[i]}, expected ${localV[i]}}"
verifyEnd = System.currentTimeMillis()
//@gcFinish
println "finished"
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"

