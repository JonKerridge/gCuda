package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class FusedMltplyAdd {
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
def fusedMultiplyAdd = {float[] a, float[] b, float[] c, float[] d, int n ->
  int row = blockIdx.y * blockDim.y + threadIdx.y
  int col = blockIdx.x * blockDim.x + threadIdx.x
  if (row < n && col < n) {
    float sum = 0.0f
    for ( int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * n + col]
    }
    d[row * n + col] = sum + c[row * n + col]
  }
} // fusedMultiplyAdd

//@gcDataToGPU
int mSize = 2   // needs to be much larger to make use of a GPU
float[] a = new float[mSize * mSize]  // matrices stored linearly in ROW order
float[] b = new float[mSize * mSize]
float[] c = new float[mSize * mSize]

//@gcDataFromGPU
float[] d = new float[mSize * mSize]

//@gcDataBoth

//@gcDataInitialise
for ( i in 0 ..< mSize * mSize){
  a[i] = (float) i
  b[i] = (float) i
  c[i] = (float) i
  d[i] = 0.0f
}

blockSize.x = 16
gridSize.x = (int)Math.ceil((double) mSize / blockSize.x)
blockSize.y = 16
gridSize.y = (int)Math.ceil((double) mSize / blockSize.y)
println " blockSize = $blockSize, gridSize = $gridSize, mSize = $mSize"
gpuStart = System.currentTimeMillis()

//@gcKernelParams a,b,c,d,mSize
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts/nFusedMltplyAdd.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'fusedMultiplyAdd')
		CUdeviceptr paramToGPU1 = new CUdeviceptr()
		cuMemAlloc (paramToGPU1, a.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU1, Pointer.to(a), a.size() * Sizeof.FLOAT)
		CUdeviceptr paramToGPU2 = new CUdeviceptr()
		cuMemAlloc (paramToGPU2, b.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU2, Pointer.to(b), b.size() * Sizeof.FLOAT)
		CUdeviceptr paramToGPU3 = new CUdeviceptr()
		cuMemAlloc (paramToGPU3, c.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU3, Pointer.to(c), c.size() * Sizeof.FLOAT)
		CUdeviceptr paramFromGPU4 = new CUdeviceptr()
		cuMemAlloc (paramFromGPU4, d.size() * Sizeof.FLOAT)
		int[] paramToGPU0 = new int[]{mSize} 
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU1),
			Pointer.to( paramToGPU2),
			Pointer.to( paramToGPU3),
			Pointer.to( paramFromGPU4),
			Pointer.to( paramToGPU0) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(d), paramFromGPU4, d.size() * Sizeof.FLOAT)

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()

//@gcEmulate
blockDim = blockSize

float[] ld = new float[mSize*mSize]
for ( gy in 0 ..< gridSize.y) {
  blockIdx.y = gy
  for (gx in 0 ..< gridSize.x) {
    blockIdx.x = gx
    for (ty in 0 ..< blockSize.y){
      threadIdx.y = ty
      for ( tx in 0 ..< blockSize.x){
        threadIdx.x = tx
        fusedMultiplyAdd(a, b, c, ld, mSize)
      }
    }
  }
}
emulateEnd = System.currentTimeMillis()

//@gcFinalise

d = [2, 4, 8, 14]   //for local testing comment out when using GPU
for ( i in 0 ..< mSize* mSize){
  assert Math.abs(ld[i] - d[i]) < 1e-5 :
      "At index $i found ${d[i]} but expected ${ld[i]}"
}
verifyEnd = System.currentTimeMillis()

//@gcFinish
		cuMemFree(paramToGPU1)
		cuMemFree(paramToGPU2)
		cuMemFree(paramToGPU3)
		cuMemFree(paramFromGPU4)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"
  }
}
