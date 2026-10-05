package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class FloatAXPY {
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
def faxpy = { int n, float a, float[] x, float[] y ->
  int i = blockIdx.x*blockDim.x + threadIdx.x
  if (i < n) y[i] = a*x[i] + y[i]
}

//@gcDataToGPU
int n = 1000000
float a = 2.0f
float[] x = new float[n]

//@gcDataFromGPU

//@gcDataBoth
float[] y = new float[n]

//@gcDataInitialise
for ( i in 0 ..< n){
  x[i] = 1.0f
  y[i] = 2.0f
}
blockSize.x = 384
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) n / blockSize.x)
println " blockSize = $blockSize, gridSize = $gridSize, n = $n"
gpuStart = System.currentTimeMillis()

//@gcKernelParams n,a,x,y
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts//nFloatAXPY.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'faxpy')
		int[] paramToGPU0 = new int[]{n} 
		float[] paramToGPU1 = new float[]{a} 
		CUdeviceptr paramToGPU2 = new CUdeviceptr()
		cuMemAlloc (paramToGPU2, x.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU2, Pointer.to(x), x.size() * Sizeof.FLOAT)
		CUdeviceptr paramBoth3 = new CUdeviceptr()
		cuMemAlloc (paramBoth3, y.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramBoth3, Pointer.to(y), y.size() * Sizeof.FLOAT)
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU0),
			Pointer.to( paramToGPU1),
			Pointer.to( paramToGPU2),
			Pointer.to( paramBoth3) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(y), paramBoth3, y.size() * Sizeof.FLOAT)

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()

//@gcEmulate

blockDim = blockSize
float[] localy = new float[n]
for ( i in 0 ..< n) localy[i] = 2.0f
for (gx in 0 ..< gridSize.x) {
  blockIdx.x = gx
  for (tx in 0..<blockSize.x) {
    threadIdx.x = tx
    faxpy(n, a, x, localy)
  }
}

emulateEnd = System.currentTimeMillis()

//@gcFinalise
for ( i in 0 ..< n)
  assert Math.abs(y[i] - 4.0f) < 1e-5 :
    "At index $i got ${y[i]}, expected 4.0}"

verifyEnd = System.currentTimeMillis()
//@gcFinish
		cuMemFree(paramToGPU2)
		cuMemFree(paramBoth3)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


  }
}
