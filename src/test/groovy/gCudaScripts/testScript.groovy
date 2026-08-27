import gCuda.Dim3

//@gCudaDriverKernel
    Dim3 blockDim = new Dim3()
    Dim3 blockIdx = new Dim3()
    Dim3 threadIdx = new Dim3()
    int sharedMemoryBytes = 0
    Long hStream = 0 
    List extra = null 

//@gCudaDataIn

//@gCudaDataOut

//@gCudaDataInOut

// add the parameter variables, comma separated,
//in the order they appear in the kernel definition 
//to the end of the following line
//@gCudaKernelParams 

// insert the Cuda Kernel definition(s) as a groovy closure of the form below
// where a,b, etc are parameters whose type must be specified and must match
// the list of parameters specified in the //@gCudakernalparams above
// def kName = { <T> a, <T> b ,  ... ->
//   closure content which will be copied into the C definition
// }//@gCudaKernel<

//@>

// initialise the host data

// add the kernel code kName to end of following line
//@gCudaLaunchKernel

//@gCudaEndKernel

// if required emulate the whole process

// now add any final host coding


