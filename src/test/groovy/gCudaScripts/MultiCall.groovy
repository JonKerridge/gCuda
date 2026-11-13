package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class MultiCall {
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
	
	
	//@gcKernelDefinition C
	/*@gc start of C code
	extern "C"
	__global__ void add(int n, int *a, int *b, int *sum)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i<n)
	    {
	        sum[i] = a[i] + b[i];
	    }
	
	}
	
	extern "C"
	__global__ void scalerX(int n, int *a, int s)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i<n)
	    {
	        a[i] = a[i] * s;
	    }
	
	}
	
	end of C coding */

	//@gcDataToGPU
	int elements = 1024
	int scaler = 2
	int[] a = new int[elements]
	int[] b = new int[elements]
	
	//@gcDataBoth
	int[] s = new int[elements]
	
	//@gcDataInitialise
	for ( i in 0 ..< elements){
	  a[i] = 1
	  b[i] = 1
	  s[i] = 0
	}
	
	blockSize.x = 256  // modify as required
	// determine number of blocks in grid, modify following as required
	gridSize.x = (int)Math.ceil((double) elements / blockSize.x)
	
	//@gcKernelInit
	setExceptionsEnabled(true)
	cuInit(0)
	CUdevice device = new CUdevice() 
	cuDeviceGet(device, 0) 
	CUcontext context = new CUcontext()
	cuCtxCreate(context, 0, device)
	// Load the PTX file containing the kernel 
	CUmodule module = new CUmodule() 
	cuModuleLoad(module, 'src/test/groovy/gCudaScripts/MultiCall.ptx') 
	
	CUfunction function = new CUfunction()
	cuModuleGetFunction(function, module, 'add')
	int[] paramToGPU0 = new int[]{elements} 
	CUdeviceptr paramToGPU2 = new CUdeviceptr()
	cuMemAlloc (paramToGPU2, a.size() * Sizeof.INT)
	cuMemcpyHtoD (paramToGPU2, Pointer.to(a), a.size() * Sizeof.INT)
	CUdeviceptr paramToGPU3 = new CUdeviceptr()
	cuMemAlloc (paramToGPU3, b.size() * Sizeof.INT)
	cuMemcpyHtoD (paramToGPU3, Pointer.to(b), b.size() * Sizeof.INT)
	CUdeviceptr paramBoth4 = new CUdeviceptr()
	cuMemAlloc (paramBoth4, s.size() * Sizeof.INT)
	cuMemcpyHtoD (paramBoth4, Pointer.to(s), s.size() * Sizeof.INT)
	Pointer kernelParameters = new Pointer()
	kernelParameters = Pointer.to (
		Pointer.to( paramToGPU0),
		Pointer.to( paramToGPU2),
		Pointer.to( paramToGPU3),
		Pointer.to( paramBoth4) 
	)
	cuLaunchKernel ( function,
		gridSize.x, gridSize.y, gridSize.z,
		blockSize.x, blockSize.y, blockSize.z,
		sharedMemoryBytes, hStream,
		kernelParameters, extra
		)
	cuCtxSynchronize()
	cuMemcpyDtoH(Pointer.to(s), paramBoth4, s.size() * Sizeof.INT)
	
	cuModuleGetFunction(function, module, 'scalerX')
	cuMemAlloc (paramBoth4, s.size() * Sizeof.INT)
	cuMemcpyHtoD (paramBoth4, Pointer.to(s), s.size() * Sizeof.INT)
	int[] paramToGPU1 = new int[]{scaler} 
	kernelParameters = Pointer.to (
		Pointer.to( paramToGPU0),
		Pointer.to( paramBoth4),
		Pointer.to( paramToGPU1) 
	)
	cuLaunchKernel ( function,
		gridSize.x, gridSize.y, gridSize.z,
		blockSize.x, blockSize.y, blockSize.z,
		sharedMemoryBytes, hStream,
		kernelParameters, extra
		)
	cuCtxSynchronize()
	cuMemcpyDtoH(Pointer.to(s), paramBoth4, s.size() * Sizeof.INT)
	//@gcKernelEnd
	
	//@gcEmulate
	
	blockDim = blockSize
	
	int[] localS = new int[elements]
	for ( i in 0 ..< elements)
	  localS[i] = (a[i] + b[i]) * scaler
	
	//@gcFinalise
	for (i in 0 ..< elements)
	   assert localS[i] == s[i]
	
	//@gcFinish
	cuMemFree(paramToGPU2)
	cuMemFree(paramToGPU3)
	cuMemFree(paramBoth4)
	println "Finished"
	
	
	} // main
} // MultiCall
