package ExamplesFromTheWeb

import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

/*
This is a groovy program written directly from the 
JCudaVectorAdd.java program which is known to work.
This is to enure that a basic groovy code works.

It has been augmented with annotations that will guide code
generation from an executable groovy console script
 */
class GcudaAdd {

  static void main(String[] args) {
//@gcDriverKernel  path appName
    // data structures used by the kernel launcher
    Dim3 gridSize = new Dim3()
    Dim3 blockSize = new Dim3()  //must be multiple of 32
    int sharedMemBytes = 0
    CUstream hStream = null
    Pointer extra = null
    // data structures used in the definition of the kernel
    // may also be used in any emulation code if present
    Dim3 blockIdx = new Dim3()
    Dim3 blockDim = new Dim3()
    Dim3 threadIdx = new Dim3()
    // the path name of the ptx file containing the compiled kernel code
    String ptxFileName = "src/main/groovy/jonKerridge/JCudaSampleAdd.ptx"

//@gcKernelDefinition
    //define the kernel operation as a closure
    def add = {int n, float[] a, float[] b, float[] sum ->
      int i = blockIdx.x * blockDim.x + threadIdx.x
      if (i<n) {
        sum[i] = (float) (a[i] + b[i])
      }
    } //add

//@gcDataToGPU
    int numElements = 1000000
    float[] hostInputA = new float[numElements]
    float[] hostInputB = new float[numElements]
//@gcDataFromGPU
    float[] hostOutput = new float[numElements]
//@gcDataBoth

//@gcDataInitialise
    for( i in  0 ..< numElements ) {
      hostInputA[i] = (float)i
      hostInputB[i] = (float)i
    }
    long gpuStart, gpuEnd, localEnd, verifyEnd
    gpuStart = System.currentTimeMillis()

    blockSize.x = 384  // number of GT1030 cores, multiple of 32
    gridSize.x = (int)Math.ceil((double)numElements / blockSize.x)


//@gcKernelParams numElements, hostInputA, hostInputB, hostOutput

    int[] vSize = new int[]{numElements}
    JCudaDriver.setExceptionsEnabled(true)
    // Initialize the driver and create a context for the first device.
    cuInit(0) 
    CUdevice device = new CUdevice() 
    cuDeviceGet(device, 0) 
    CUcontext context = new CUcontext() 
    cuCtxCreate(context, 0, device) 

    // Load the ptx file.
    CUmodule module = new CUmodule() 
    cuModuleLoad(module, ptxFileName) 

    // Obtain a function pointer to the "add" function.
    CUfunction function = new CUfunction() 
    cuModuleGetFunction(function, module, "add")



    // Allocate the device input data, and copy the
    // host input data to the device
    CUdeviceptr deviceInputA = new CUdeviceptr() 
    cuMemAlloc(deviceInputA, hostInputA.size() * Sizeof.FLOAT)
    cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
        hostInputA.size() * Sizeof.FLOAT)
    CUdeviceptr deviceInputB = new CUdeviceptr() 
    cuMemAlloc(deviceInputB, hostInputB.size() * Sizeof.FLOAT)
    cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
        hostInputB.size() * Sizeof.FLOAT)

    // Allocate device output memory
    CUdeviceptr deviceOutput = new CUdeviceptr() 
    cuMemAlloc(deviceOutput, hostOutput.size() * Sizeof.FLOAT)

    // Set up the kernel parameters: A pointer to an array
    // of pointers which point to the actual values.

    Pointer kernelParameters = Pointer.to(
        Pointer.to(vSize),
        Pointer.to(deviceInputA),
        Pointer.to(deviceInputB),
        Pointer.to(deviceOutput)
    ) 

    // Call the kernel function.
    cuLaunchKernel(function,
        gridSize.x,  gridSize.y, gridSize.z,      // Grid dimension
        blockSize.x, blockSize.y, blockSize.z,      // Block dimension
        sharedMemBytes,               // Shared memory size
        hStream,                      // CUstream, typically null
        kernelParameters,  // Kernel parameters
        extra
    ) 
    cuCtxSynchronize() 

    // copy the device output to the host memory.

    cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
        hostOutput.size() * Sizeof.FLOAT)
//@gcKernelEnd
    gpuEnd = System.currentTimeMillis()

//@gcEmulate
    // emulate GPU operation
    blockDim.x = blockSize.x
    blockDim.y = blockSize.y
    blockDim.z = blockSize.z

    float[] localOutput = new float[numElements]

    for (x in 0 ..< gridSize.x){
      blockIdx.x = x
      for (t in 0 ..< blockDim.x) {
        threadIdx.x = t
        add (numElements, hostInputA, hostInputB, localOutput)
      }
    }
    // check the following test works
//    localOutput[0] = 10.0f  // overwrite with wrong value
//@gcFinalise
    localEnd = System.currentTimeMillis()

    // Verify the result
    boolean passed = true 
    for( i in  0 ..< numElements)
    {
      float expected = i+i 
//      println "local $i = $localOutput[i]"
      if ((Math.abs(hostOutput[i] - expected) > 1e-5) ||
          (Math.abs(hostOutput[i] - localOutput[i]) >1e-5) )
      {
        println "At index $i found ${localOutput[i]} but expected ${hostOutput[i]}"
        passed = false 
        break 
      }
    }
    verifyEnd = System.currentTimeMillis()
    println "Test ${(passed?'PASSED':'FAILED')}"
    println "GPU time = ${gpuEnd - gpuStart}"
    println "Local time = ${localEnd - gpuEnd}"
    println "Verify time = ${verifyEnd - localEnd}"

//@gcFinish
    // Clean up.
    cuMemFree(deviceInputA) 
    cuMemFree(deviceInputB) 
    cuMemFree(deviceOutput) 

  } //main
  
} // class
