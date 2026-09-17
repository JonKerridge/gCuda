package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class VectorAdd {
	static void main(String[] args) {

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
    def add = { int n, float[] a, float[] b, float[] sum ->
      int i = blockIdx.x * blockDim.x + threadIdx.x
      if (i < n) {
        sum[i] = (float) (a[i] + b[i])
      }
    } //add


//@gcDataToGPU
    int vectorSize = 100  // needs to be at least 10000000 to show GPU speedup
    float[] vectorA = new float[vectorSize]
    float[] vectorB = new float[vectorSize]

//@gcDataFromGPU

    float[] outVector = new float[vectorSize]

//@gcDataBoth

//@gcDataInitialise
    for (i in 0..<vectorSize) {
      vectorA[i] = (float) i
      vectorB[i] = (float) i
    }
    blockSize.x = 32
    gridSize.x = (int) Math.ceil((double) vectorSize / blockSize.x)

    println " blockSize = $blockSize, gridSize = $gridSize, vectorSize = $vectorSize"
    // determines number of blocks in grid
    gpuStart = System.currentTimeMillis()

//@gcKernelParams vectorSize, vectorA, vectorB, outVector
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts/nVectorAdd.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'add')
		int[] paramToGPU0 = new int[]{vectorSize} 
		CUdeviceptr paramToGPU1 = new CUdeviceptr()
		cuMemAlloc (paramToGPU1, vectorA.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU1, Pointer.to(vectorA), vectorA.size() * Sizeof.FLOAT)
		CUdeviceptr paramToGPU2 = new CUdeviceptr()
		cuMemAlloc (paramToGPU2, vectorB.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU2, Pointer.to(vectorB), vectorB.size() * Sizeof.FLOAT)
		CUdeviceptr paramFromGPU3 = new CUdeviceptr()
		cuMemAlloc (paramFromGPU3, outVector.size() * Sizeof.FLOAT)
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU0),
			Pointer.to( paramToGPU1),
			Pointer.to( paramToGPU2),
			Pointer.to( paramFromGPU3) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(outVector), paramFromGPU3, outVector.size() * Sizeof.FLOAT)

//@gcKernelEnd
    gpuEnd = System.currentTimeMillis()

//@gcEmulate
blockDim = blockSize

    float[] localOutput = new float[vectorSize]

    for (x in 0..<gridSize.x) {
      blockIdx.x = x
      for (t in 0..<blockDim.x) {
        threadIdx.x = t
        add(vectorSize, vectorA, vectorB, localOutput)
      }
    }
    emulateEnd = System.currentTimeMillis()

//@gcFinalise
    for (i in 0..< vectorSize) {
//      assert (Math.abs(localOutput[i] - 2*i) < 1e-5) :
      assert (Math.abs(localOutput[i] - outVector[i]) < 1e-5) :
        "At index $i found ${localOutput[i]} but expected ${outVector[i]}"
    }
    verifyEnd = System.currentTimeMillis()
//@gcFinish
		cuMemFree(paramToGPU1)
		cuMemFree(paramToGPU2)
		cuMemFree(paramFromGPU3)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"

  }
}
