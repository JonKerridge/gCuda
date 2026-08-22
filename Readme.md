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

In the Groovy script I have introduced @ annotations preceded by // so that they do not cause
any problem when compiling the script and running it. Obviously in a functioning system, the // would 
be removed so the Groovy system could process the annotations.<br>

The vectorAdd and vectorScale scripts show the minimal Groovy coding required with the annotations added.<br>

CUDA assumes the existence of some accessible values that are used to access memory.  These have to
be included in the Groovy script.  They have been included with an _ separator rather than
a period as that would break Groovy naming rules.  There are other default CUDA variables that
can be accessed by kernel code and we should include them all by default when creating such a script.<br>

The next thing we need to do is define the data that will be passed between host and GPU.  I have introduced
three annotations to specify the data; _@gCudaDataIn_, @_gCudaDataOut_ and @_gCudaDataInOut_.  The first is for data 
being transferred from the host to the GPU before the kernel is invoked,
the second for data being transferred from the GPU to the host once kernel processing has finished, and finally an option in which
data is transferred to the GPU from the host at the start of kernel processing and is then transferred back to
the host once kernel proceesing is finished.

If we consider the result of generating the required code once that annotated Groovy script has been processed
and the aim is to generate Java then the above annotations will generate the code contained in 
lines 82-95 and 119-121 of JCudaVectorAdd_java.

The annotation _@gCudaKernel<_ is used to define the kernel operations that are to be undertaken.  I think it is possible to define
several such methods so they are terminated by _@>_.

The result of processing the Groovy specification will be a separate file for the operations with
the variable names modified to what is expected.  This will lead to tge generation of lines
52-69 and 97-104 of the java code.

The script then does some real processing by initialising the required data and defining the values for kernel parameters
such a block and grid size.

The annotation @gCudaLaunchKernel ...   causes the call of the kernel function
and results in the generation of lines 109-115 in the java code.

The groovy script now contains a doe sequence that emulates the function on the kernel and this would not be required in the 
resulting Java code.  This part of the script is terminated by the _@gCudaEndKernel_ annotation.

Finally, the script contains any code required once the kernel function has completed.

##### Commentary
The approach I have adopted in the first instance is to create an environment in which a Groovy programmer can write a single script that
contains all the code necessary for both the processing on the host and the GPU.  This has the benefit
of documenting the whole operation in a single place.  I have chosen the route of using annotations in the script as this is a very
simple way of identifying the prupose of each part of the script.  A major benefit of this approach is the whole processing from start
to finish can be tested because the effect of the GPU operation can be emulated in the 
Groovy script.  Needless to say the Groovy script approach is much less verbose that the equivalent Java and C coding, 64 lines as 
opposed to a total of 156 lines, spread over two files.

The only drawback is that in the approach I have described the standard names assocaited with CUDA implicit variables has had to be changed
for those that use a period separator.  However, with this change it is then possible to provide an emulation of the final host/GPU system.

It should be noted that I have not yet written any CUDA code and I am only now just getting to grips with the enormity of the task.
I have to look at how the extensive libraries are used and invoked but the jCuda website has many examples.  I have yet to find an example which processes 
multidimensional matrices that I can use as a model for further exploration of the approach.

I am also just getting to grips with the structure of GPUs and the somewhat preculair way they are
structured internally and the somewhat bizzare way user code can refer to variables
that are supplied by the kernel.  This is all probably very obvious but I just needed to say it in 
case you thought I was a CUDA expert.