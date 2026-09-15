package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class MatrixMultiply {
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
def matrixMultiply = {float[] A, float[] B, float[] C, int N ->
  int row = blockIdx.y * blockDim.y + threadIdx.y
  int col = blockIdx.x * blockDim.x + threadIdx.x
  if (row < N && col < N) {
    float sum = 0.0f
    for ( int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * N + col]
    }
    C[row * N + col] = sum
  }
} //matrixMultiply

//@gcDataToGPU
int mSize = 200
float[] a = new float[mSize * mSize]  // matrices stored linearly in ROW order
float[] b = new float[mSize * mSize]      // only one property per line



//@gcDataFromGPU
float[] c = new float[mSize * mSize]

//@gcDataBoth

//@gcDataInitialise

for ( i in 0 ..< mSize * mSize){
  a[i] = (float) i
  b[i] = (float) i
  c[i] = 0.0f
}

blockSize.x = 16   // must be a multiple of 32, depends on GPU used
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) mSize / blockSize.x)
blockSize.y = 16
gridSize.y = (int)Math.ceil((double) mSize / blockSize.y)
println " blockSize = $blockSize, gridSize = $gridSize, mSize = $mSize"
gpuStart = System.currentTimeMillis()

//@gcKernelParams a,b,c,mSize
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts/nMatrixMultiply.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'matrixMultiply')
		CUdeviceptr paramToGPU1 = new CUdeviceptr()
		cuMemAlloc (paramToGPU1, a.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU1, Pointer.to(a), a.size() * Sizeof.FLOAT)
		CUdeviceptr paramToGPU2 = new CUdeviceptr()
		cuMemAlloc (paramToGPU2, b.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU2, Pointer.to(b), b.size() * Sizeof.FLOAT)
		CUdeviceptr paramFromGPU3 = new CUdeviceptr()
		cuMemAlloc (paramFromGPU3, c.size() * Sizeof.FLOAT)
		int[] paramToGPU0 = new int[]{mSize} 
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU1),
			Pointer.to( paramToGPU2),
			Pointer.to( paramFromGPU3),
			Pointer.to( paramToGPU0) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(c), paramFromGPU3, c.size() * Sizeof.FLOAT)

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()
//@gcEmulate
blockDim.x = blockSize.x
blockDim.y = blockSize.y
blockDim.z = blockSize.z

float[] localOutput = new float[mSize*mSize]
for (row in 0 ..< mSize){
  for (col in 0 ..< mSize){
    float cVal = 0.0f
    for ( k in 0 ..< mSize){
      cVal = cVal + a[row * mSize + k] * b[k * mSize + col]
    }
    localOutput[row * mSize + col] = cVal
  }
}
emulateEnd = System.currentTimeMillis()
//@gcFinalise
println "finished"
boolean passed = true
for ( i in 0 ..< mSize* mSize){
  if (Math.abs(localOutput[i] - c[i]) > 1e-5) {
    println "At index $i found ${localOutput[i]} but expected ${c[i]}"
    passed = false
    break
  }
}
println "Test ${(passed ? 'PASSED' : 'FAILED')}"

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
