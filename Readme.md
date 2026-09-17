This is a first attempt at adding annotations to a functional Groovy script to invoke
some of the computation on a CUDA GPU.
The examples are based on examples obtained from the JCuda.org website and GitHub<br>
http://www.jcuda.org/tutorial/TutorialIndex.html <br>
https://github.com/jcuda/jcuda-samples <br>
specifically I have copied <br>
https://github.com/jcuda/jcuda-samples/blob/master/JCudaSamples/src/main/resources/kernels/JCudaVectorAddKernel.cu <br>
and<br>
https://github.com/jcuda/jcuda-samples/JCudaSamples/src/main/java/jcuda/driver/samples/JCudaVectorAdd.java <br>

into this project to make comparison easier to assimilate as simple text files.

Observation of the java calling code shows the amount of boilerplate required! <br>

In the Groovy script I have introduced @ annotations preceded by // so that they do 
not cause any problem when compiling the script and running it.<br>

The  scripts show the minimal Groovy coding required with the annotations added.<br>

The _//@gcDriverKernel_ annotation indicates that the application is using a kernel
and driver mechanism, rather than pre-written libraries.  The path specifies where the
resultant class is to be saved and the name indicates the name of the class.  
The GPUcompute specifies the compute architecture of the GPU being used currently;
Pascal 61, Turing 75, Ampere 86 Ada 89 and Blackwell 120.


CUDA assumes the existence of some accessible values that are used to access memory.  
These have to be included in the Groovy script.  There are other default CUDA variables 
that can be accessed by kernel code, and we include them all by default when 
creating such a script.


The annotation _@gKernelDefintion_ is used to define the kernel operations that are
to be undertaken.  It takes the form of a closure which will be converted into  a
Cuda .cu file.  The parameters are specified using a Java and C style declaration.

The next thing we need to do is define the data that will be passed between host and GPU.  
I have introduced three annotations to specify the data; _@ggcDataToGPU_, _@gcDataFromGPU_ 
and @_gcDataBoth_.  The first is for data being transferred from the host to the GPU 
before the kernel is invoked, the second for data being transferred from the GPU to the 
host once kernel processing has finished. Finally, an option in which data is transferred 
to the GPU from the host at the start of kernel processing and is then transferred back to
the host once kernel processing is finished.

NOTE the declarations uses a Java and C style declaration so they can be more easily accessed in
the resulting code.  The use of the Groovy _List_ mechanism would have introduced a lot of 
internal data transfers from arrayList to array.

The aim is to generate a Groovy class using the above annotations to generate the GPU 
access code.

The _//@gcDataInitialse_ annotation specifies the initialisation of all the required 
variables being passed between CPU and GPU and vice-versa.

The result of processing the Groovy specification will be a separate file for the operations 
with the variable names modified to what is expected.  

The script then does some real processing by initialising the required data and defining 
the values for kernel parameters such a block and grid size.

The annotation _//@gcKernelParams ..._   causes the call of the kernel function with the 
application parameters specified in the order they are required by the kernel method.

The annotation _//@gcKernelEnd_ signifies the  place where any following code will be processed
as inline Groovy code.


The _//@gcEmulate_ annotation indicates where any user code that emulates the required 
processing usinf the CPU only can be written.

The annotation _//@gcFinalise_ indicates that any following coding will be executed inline 
and can be used to check the processing by the CPU and the CPU emulated code as identical.
The example scripts have two assert statements defined; one for when doing local testing 
and the other when the code has been built and is using a GPU.  Only one should be used at
any one time.  

The _//@gcFinalise_ introduces any inline code that can be used to, for example, print 
out any timing data.  The scripts in src/test/groovy/gCudaScripts all contain 
additional coding to calculate and print the times associated with each phase of 
the application.

There are example scripts and built files for Pascal architecture based GPU cards in the 
src/test/groovy/gCudaScripts folder, for Scaler multiply, vector add, symmetric matrix multiply,
 asymmetric matrix multiply, and fused multiply add.

###### IMPORTANT NOTE
All the scripts have vector and matrix sizes that make them amenable to testing on a
normal CPU and thus do not show any speed up when processed on a GPU.  The sizes have
to be increased substantially, for that to be seen.  The benefit of the script approach
is that the user can be confident the kernel definition will work when executed
on a GPU as the kernel have been exercised in exactly the same way as will occur in a GPU.

##### Commentary
The approach I have adopted in the first instance is to create an environment in which a 
Groovy programmer can write a single script that contains all the code necessary for both 
the processing on the host and the GPU.  This has the benefit of documenting the whole 
operation in a single place.  I have chosen the route of using annotations in the 
script as this is a very simple way of identifying the purpose of each part of the script.
A major benefit of this approach is the whole processing from start to finish can be 
tested because the effect of the GPU operation can be emulated in the Groovy script.  
Needless to say the Groovy script approach is much less verbose that the equivalent 
Java and C coding, 64 lines as opposed to a total of 156 lines, spread over two files.

I am also just getting to grips with the structure of GPUs and the somewhat peculiar 
way they are structured internally and the somewhat bizarre way user code can refer 
to variables that are supplied by the kernel.  This is all probably very obvious but,
I just needed to say it in case you thought I was a CUDA expert.