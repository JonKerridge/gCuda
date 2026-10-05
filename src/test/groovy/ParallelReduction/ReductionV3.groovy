package ParallelReduction

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class ReductionV3 {
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

int sizeOfBlock = 128
int elements = 96 * sizeOfBlock   // number of elements in the input array
// 96 is limit of size for GT1030

//@gcKernelDefinition
def reductionV3 = { int[] inData, int[] outData ->
  //@gc:extern __shared__ int sData[]
  // thread loads one elements from inData to shared sData
  int tid = threadIdx.x
  int i = blockIdx.x * blockDim.x + threadIdx.x
  sData[tid] = inData[i]
  //@gc:__syncthreads()
  // do reduction in shared sData
  for ( int s = blockDim.x / 2; s > 0; s >>= 1){
    if (tid < s)
      sData[tid] += sData[tid + s]
    //@gc:__syncthreads()
  }
  // write result back to outData
  if (tid == 0) outData[blockIdx.x] = sData[0]
}


//@gcDataToGPU
int[] inData = new int[elements]

//@gcDataFromGPU
int[] outData = new int[elements/sizeOfBlock] // we assume a block size of 128

//@gcDataBoth

//@gcDataInitialise
Random random = new Random()
for ( i in 0 ..< elements) inData[i] = random.nextInt(20)

blockSize.x = sizeOfBlock
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) elements / blockSize.x)
sharedMemoryBytes = elements * Sizeof.INT

println " blockSize = $blockSize, gridSize = $gridSize, elements = $elements"
gpuStart = System.currentTimeMillis()

//@gcKernelParams inData, outData
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/ParallelReduction//nReductionV3.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'reductionV3')
		CUdeviceptr paramToGPU0 = new CUdeviceptr()
		cuMemAlloc (paramToGPU0, inData.size() * Sizeof.INT)
		cuMemcpyHtoD (paramToGPU0, Pointer.to(inData), inData.size() * Sizeof.INT)
		CUdeviceptr paramFromGPU1 = new CUdeviceptr()
		cuMemAlloc (paramFromGPU1, outData.size() * Sizeof.INT)
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU0),
			Pointer.to( paramFromGPU1) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(outData), paramFromGPU1, outData.size() * Sizeof.INT)

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()

//@gcEmulate

blockDim = blockSize
int sum = 0
for (i in 0 ..< elements) sum = sum + inData[i]

emulateEnd = System.currentTimeMillis()

//@gcFinalise

int reductionSum = 0
for ( i in 0 ..< outData.size())
  reductionSum = reductionSum + outData[i]
assert sum == reductionSum:
  "Sums do not match: local sum = $sum, reduction sum = $reductionSum"
verifyEnd = System.currentTimeMillis()

//@gcFinish
		cuMemFree(paramToGPU0)
		cuMemFree(paramFromGPU1)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


  }
}
