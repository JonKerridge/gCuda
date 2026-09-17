package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class ScalerMultiply {
	static void main(String[] args) {

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
    vector[i] = (float) (vector[i] * s)
  }
} // scalerMultiply

//@gcDataToGPU
int s = 2
int vSize = 2000  // needs to be much larger to show GPU speedup
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
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts/nScalerMultiply.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'scalerMultiply')
		int[] paramToGPU0 = new int[]{s} 
		int[] paramToGPU1 = new int[]{vSize} 
		CUdeviceptr paramBoth2 = new CUdeviceptr()
		cuMemAlloc (paramBoth2, vector.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramBoth2, Pointer.to(vector), vector.size() * Sizeof.FLOAT)
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU0),
			Pointer.to( paramToGPU1),
			Pointer.to( paramBoth2) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(vector), paramBoth2, vector.size() * Sizeof.FLOAT)

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()
//@gcEmulate
blockDim = blockSize

float[] localV = new float[vSize]
for ( i in 0 ..< vSize) localV[i] =  (float)i

for (gx in 0 ..< gridSize.x) {
  blockIdx.x = gx
  for (tx in 0..<blockSize.x) {
    threadIdx.x = tx
    scalerMultiply(s, vSize, localV)
  }
}
emulateEnd = System.currentTimeMillis()
//@gcFinalise
for ( i in 0 ..<vSize)
//  assert Math.abs(localV[i] - 2*i) < 1e-5 : used for initial kernel checking
  assert Math.abs(localV[i] - vector[i]) < 1e-5 :
      "At index $i got ${vector[i]}, expected ${localV[i]}}"
verifyEnd = System.currentTimeMillis()
//@gcFinish
		cuMemFree(paramBoth2)
println "finished"
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"
  }
}
