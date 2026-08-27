Run the CreateDriverScript and specify the name of the script
when requested.  An empty script will be placed in the directory specified,
currently src/test/groovy/gCudaScripts

Open a Groovy Console and copy the script into the console area

Modify the script to undertake the required application.

If you add code to emulate the operation as if it were run on a GPU
ensure you reduce the size of the application to sensible proportions.

Once you have got it working you can then copy the updated code back into
script file overwriting what was there.

you can now convert this to a jCuda application using the TransformScript program,
which will generate a jCuda program that will run on a host with GPU.  The 
program will be placed in the src/test/gtoovy/jCudaScripts folder