Run InitialScript and specify the path and name of the script
when requested.  An empty script will be placed in the directory specified with
path/nameScript.groovy name.  
In the following this is referred to as _**filename**_.  This file will be of type .groovy.
InitialScript can be executed as a main script, where you type in the required 
values or can be invoked using the method createScript().

Open a Groovy Console and copy the script into the console area

Modify the script to undertake the required application.

If you add code to emulate the operation, as if it were run on a GPU,
ensure you reduce the size of the application to sensible proportions. The
example scripts show how the kernel definition cam be emulated as if
theu were running on a GPU using CUDA internal idiomatic vector and matrix addressing.

Once you have got it working you can then copy the updated code back into
script file overwriting what was there originally.

You can now convert this to a groovyCuda application using the Builder program,
which will generate a gCuda program that will run on a host with GPU.  The 
program will be placed in the same folder with the name _fileName.groovy_ which is specified in
the _//@gcDriverKernel_ annotation along with folder in which the file is to be stored.  You also specify
the compute capability of the GPU card being used.  For example Pascal GPUs have a computer capability of
61, Turing 75, Ampere 96, Ada 89 and Blackwell 120.

Builder also generates a file with a similar name, starting with 'c'
and type '.cu' as **c*filename*.cu** .

Builder is invoked using the build() method.  The required script can be typed in or the file name 
coordinates can be passed as a pair of parameters.

This file is then compiled using the installed version of NVIDIA nvcc (c compiler).
The compilation will create a file called **n*filename*.ptx**.

Finally, the Groovy class file,  _fileName.groovy_ , can be executed
and should execute on your host PC and GPU.

There are several examples in src/test/groovy/gCudaScripts, VectorAdd, MatrixMultiply,
AsymMatrixMultiply, and ScalerMultiply.