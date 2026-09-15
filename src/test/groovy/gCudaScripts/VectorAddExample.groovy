

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

//@gcDriverKernel   src/test/groovy/gCudaScripts  VectorAddEG

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
    int vectorSize = 1000000
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
    blockSize.x = 32   // must be a multiple of 32, depends on GPU used
    gridSize.x = (int) Math.ceil((double) vectorSize / blockSize.x)
    // determines number of blocks in grid
    gpuStart = System.currentTimeMillis()

//@gcKernelParams vectorSize, vectorA, vectorB, outVector

//@gcKernelEnd
    gpuEnd = System.currentTimeMillis()

//@gcEmulate
    blockDim.x = blockSize.x
    blockDim.y = blockSize.y
    blockDim.z = blockSize.z

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
    boolean passed = true
    for (i in 0..< vectorSize) {
      if (Math.abs(localOutput[i] - outVector[i]) > 1e-5) {
        println "At index $i found ${localOutput[i]} but expected ${outVector[i]}"
        passed = false
        break
      }
    }
    println "Test ${(passed ? 'PASSED' : 'FAILED')}"
    verifyEnd = System.currentTimeMillis()
//@gcFinish
println "Data initialise : ${gpuStart-startTime} msecs"
println "GPU run time    : ${gpuEnd-gpuStart} msecs"
println "Emulate time    : ${emulateEnd-gpuEnd} msecs"
println "Verify time     : ${verifyEnd-emulateEnd} msecs"

