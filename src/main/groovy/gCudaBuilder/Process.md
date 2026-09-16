Run the CreateDriverScript and specify the name of the script
when requested.  An empty script will be placed in the directory specified, which
currently defaults to src/test/groovy/gCudaScripts.  In the following this is referred to as 
_**filename**_.  This file will be of type .groovy.

A good idea is to include the string _Script_ as the last part of the fileName.

Open a Groovy Console and copy the script into the console area

Modify the script to undertake the required application.

If you add code to emulate the operation, as if it were run on a GPU,
ensure you reduce the size of the application to sensible proportions. The
example scripts show how the kernel definition cam be emulated as if
theu were running on a GPU using CUDA internal idiomatic vector and matrix addressing.

Once you have got it working you can then copy the updated code back into
script file overwriting what was there originally.

You can now convert this to a groovyCuda application using the BuildApplication program,
which will generate a gCuda program that will run on a host with GPU.  The 
program will be placed in the default folder with the name _fileName.groovy_ which is specified in
the _//@gcDriverKernel_ annotation along with folder in which the file is to be stored.  You also specify
the compute capability of the GPU card being used.  For example Pascal GPUs have a computer capability of
61, Turing 75, Ampere 96, Ada 89 and Blackwell 120.

BuildApplication also generates a file with a similar name, starting with 'c'
and type '.cu' as **c*filename*.cu** .

This file is then compiled using the installed version of NVIDIA nvcc (c compiler).
The compilation will create a file called **n*filename*.ptx**.

Finally, the Groovy class file,  _fileName.groovy_ , can be executed
and should execute on your host PC and GPU.

There are several examples in src/test/groovy/gCudaScripts, VectorAdd, MatrixMultiply,
AsymMatrixMultiply, and ScalerMultiply.