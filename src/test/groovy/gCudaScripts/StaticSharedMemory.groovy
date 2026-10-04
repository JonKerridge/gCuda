package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class StaticSharedMemory {
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
def staticReverse = {int[] d, int n ->
  //@gc:__shared__ int s[64];
  int t = threadIdx.x
  int tr = n-t-1
  s[t] = d[t]
  //@gc:__syncthreads();
  d[t] = s[tr]
}

//@gcDataToGPU
int n = 64

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
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts//nStaticSharedMemory.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'staticReverse')
		CUdeviceptr paramBoth1 = new CUdeviceptr()
		cuMemAlloc (paramBoth1, d.size() * Sizeof.INT)
		cuMemcpyHtoD (paramBoth1, Pointer.to(d), d.size() * Sizeof.INT)
		int[] paramToGPU0 = new int[]{n} 
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramBoth1),
			Pointer.to( paramToGPU0) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(d), paramBoth1, d.size() * Sizeof.INT)

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
		cuMemFree(paramBoth1)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


  }
}
