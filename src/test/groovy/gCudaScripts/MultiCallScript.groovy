import gCuda.Dim3
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import static jcuda.driver.JCudaDriver.*

//@gcDriverKernel 61

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
/*@gc
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

*/

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

//@gcCall add ( elements,a,b,s)

//@gcCall scalerX ( elements,s, scaler)

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
println "Finished"


