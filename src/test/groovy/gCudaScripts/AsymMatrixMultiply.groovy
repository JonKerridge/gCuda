package gCudaScripts

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

class AsymMatrixMultiply {
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
def AsymMatrixMultiply = {float[] A, float[] B, float[] C, int M, int K, int N ->
  // C[r][c] = sumOverK of A[i][k] * B[k][j]
  int row = blockIdx.y * blockDim.y + threadIdx.y
  int col = blockIdx.x * blockDim.x + threadIdx.x
  if (row < M && col < N) {
    float sum = 0.0f
    for ( int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col]
    }
    C[row * N + col] = sum
  }
} //matrixMultiply

//@gcDataToGPU
int M = 2   // increase by at least a factor of 10 for GPU use
int K  = 3    // AWidth = BHeight or A columns = B rows
int N  = 4
float[] a = new float[M * K]  // matrices stored linearly in ROW order
float[] b = new float[K * N]  // only one property per line



//@gcDataFromGPU
float[] c = new float[M * N]

//@gcDataBoth

//@gcDataInitialise

for ( i in 0 ..< M * K) a[i] = (float) i
for ( i in 0 ..< K * N) b[i] = (float) i
for ( i in 0 ..< M * N) c[i] = 0.0f


blockSize.x = 32
// determine number of blocks in grid, modify following as required
gridSize.x = (int)Math.ceil((double) K*N / blockSize.x)
blockSize.y = 32
gridSize.y = (int)Math.ceil((double) M*K / blockSize.y)
println " blockSize = $blockSize, gridSize = $gridSize, M=$M, K=$K, N=$N"
gpuStart = System.currentTimeMillis()

//@gcKernelParams a, b, c, M, K ,N
		setExceptionsEnabled(true)
		cuInit(0)
		CUdevice device = new CUdevice() 
		cuDeviceGet(device, 0) 
		CUcontext context = new CUcontext()
		cuCtxCreate(context, 0, device)
		// Load the PTX file containing the kernel 
		CUmodule module = new CUmodule() 
		cuModuleLoad(module, 'src/test/groovy/gCudaScripts/nAsymMatrixMultiply.ptx') 
		CUfunction function = new CUfunction()
		cuModuleGetFunction(function, module, 'AsymMatrixMultiply')
		CUdeviceptr paramToGPU3 = new CUdeviceptr()
		cuMemAlloc (paramToGPU3, a.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU3, Pointer.to(a), a.size() * Sizeof.FLOAT)
		CUdeviceptr paramToGPU4 = new CUdeviceptr()
		cuMemAlloc (paramToGPU4, b.size() * Sizeof.FLOAT)
		cuMemcpyHtoD (paramToGPU4, Pointer.to(b), b.size() * Sizeof.FLOAT)
		CUdeviceptr paramFromGPU5 = new CUdeviceptr()
		cuMemAlloc (paramFromGPU5, c.size() * Sizeof.FLOAT)
		int[] paramToGPU0 = new int[]{M} 
		int[] paramToGPU1 = new int[]{K} 
		int[] paramToGPU2 = new int[]{N} 
		Pointer kernelParameters = Pointer.to (
			Pointer.to( paramToGPU3),
			Pointer.to( paramToGPU4),
			Pointer.to( paramFromGPU5),
			Pointer.to( paramToGPU0),
			Pointer.to( paramToGPU1),
			Pointer.to( paramToGPU2) 
		)
		cuLaunchKernel ( function,
			gridSize.x, gridSize.y, gridSize.z,
			blockSize.x, blockSize.y, blockSize.z,
			sharedMemoryBytes, hStream,
			kernelParameters, extra
			)
		cuCtxSynchronize()
		cuMemcpyDtoH(Pointer.to(c), paramFromGPU5, c.size() * Sizeof.FLOAT)

//@gcKernelEnd
gpuEnd = System.currentTimeMillis()
//@gcEmulate
blockDim.x = blockSize.x
blockDim.y = blockSize.y
blockDim.z = blockSize.z

float[] lc = new float[M*N]
for ( gy in 0 ..< gridSize.y) {
  blockIdx.y = gy
  for (gx in 0 ..< gridSize.x) {
    blockIdx.x = gx
    for (ty in 0 ..< blockSize.y){
      threadIdx.y = ty
      for ( tx in 0 ..< blockSize.x){
        threadIdx.x = tx
        AsymMatrixMultiply(a, b, lc, M, K, N)
      }
    }
  }
}

emulateEnd = System.currentTimeMillis()
//@gcFinalise

//c = [20, 23,26,29,56,68,80,92] // for initial testing, remove when using GPU

boolean passed = true
for ( i in 0 ..< M*N)
  assert Math.abs(lc[i] - c[i]) < 1e-5 :
      "At index $i found ${c[i]} but expected ${lc[i]}"

verifyEnd = System.currentTimeMillis()
//@gcFinish
		cuMemFree(paramToGPU3)
		cuMemFree(paramToGPU4)
		cuMemFree(paramFromGPU5)
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"


  }
}
