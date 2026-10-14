package ParallelReduction

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class ReductionV7 {
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

//@gcKernelDefinition C
/*@gc start of C code
extern "C"
__device__ void warpReduce(volatile int *sdata, unsigned int tid, int blockSize) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

extern "C"
__global__ void reductionV7 ( int *g_idata, int *g_odata, unsigned int n, int blockSize)
{
  extern __shared__ int sdata [];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = 0;
  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize;
  }
  __syncthreads();
  if (blockSize >= 512) {
    if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
  }
  if (tid < 32) warpReduce(sdata, tid, blockSize);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
end of C coding */

//@gcDataToGPU
int sizeOfBlock = 128
int elements = 16 * sizeOfBlock // number of elements in the input array
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

//@gcKernelParams inData, outData, elements, sizeOfBlock
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/ParallelReduction/nReductionV7.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'reductionV7')
		CUdeviceptr paramToGPU2 = new CUdeviceptr()
		cuMemAlloc (paramToGPU2, inData.size() * Sizeof.INT)
		cuMemcpyHtoD (paramToGPU2, Pointer.to(inData), inData.size() * Sizeof.INT)
		CUdeviceptr paramFromGPU3 = new CUdeviceptr()
		cuMemAlloc (paramFromGPU3, outData.size() * Sizeof.INT)
		int[] paramToGPU1 = new int[]{elements} 
		int[] paramToGPU0 = new int[]{sizeOfBlock} 
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU2),
			Pointer.to( paramFromGPU3),
			Pointer.to( paramToGPU1),
			Pointer.to( paramToGPU0) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(outData), paramFromGPU3, outData.size() * Sizeof.INT)

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
		cuMemFree(paramToGPU2)
		cuMemFree(paramFromGPU3)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


  }
}
