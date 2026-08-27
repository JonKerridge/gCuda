package gCudaBuilder




class CreateDriverScript {

  /*
  this script creates a blank script file into which the user can add the code
  required to build a code that uses the driverApi and kernel
   */

  /**
   *
   * @param args 0 is the required path to the script file, the last part is assumed to be a package name
   *             1 is the file and class name ( no extension required)
   * for example, ["../src/test/groovy/gCudaScripts/", "dummy"]"
   */
  static void main(String[] args) {
    String gCudaScriptName, gCudaScriptPath
//    This needs making more general but will work for now
    if (args == []) {
      print "Please enter the name of the script to generate:"
      gCudaScriptPath = "src/test/groovy/gCudaScripts/"
      gCudaScriptName = System.in.newReader().readLine()
    }
    else {
      gCudaScriptPath = args[0]
      gCudaScriptName = args[1]
    }
    println "creating: $gCudaScriptPath.$gCudaScriptName"
    String scriptName = gCudaScriptPath + gCudaScriptName + ".groovy"
    List<String> pathTokens = gCudaScriptPath.tokenize('/')
    String packageName = pathTokens[(pathTokens.size() - 1)]
    File scriptFile = new File (scriptName)
    PrintWriter sw= new PrintWriter(scriptFile)

    sw.println("import gCuda.Dim3\n")
//    sw.println ("Class $gCudaScriptName {")
//    sw.println ("  static void main(String[] args) {\n")
    sw.println ("//@gCudaDriverKernel\n" +
        "    Dim3 blockDim = new Dim3()\n" +
        "    Dim3 blockIdx = new Dim3()\n" +
        "    Dim3 threadIdx = new Dim3()\n" +
        "    int sharedMemoryBytes = 0\n" +
        "    Long hStream = 0 \n" +
        "    List extra = null \n"
    )

    sw.println("//@gCudaDataIn\n\n" +
        "//@gCudaDataOut\n\n" +
        "//@gCudaDataInOut\n\n" +
        "// add the parameter variables, comma separated,\n" +
        "//in the order they appear in the kernel definition \n" +
        "//to the end of the following line\n"+
        "//@gCudaKernelParams \n\n" +
        "// insert the Cuda Kernel definition(s) as a groovy closure of the form below\n" +
        "// where a,b, etc are parameters whose type must be specified and must match\n" +
        "// the list of parameters specified in the //@gCudakernalparams above\n" +
        "// def kName = { <T> a, <T> b ,  ... ->\n" +
        "//   closure content which will be copied into the C definition\n" +
        "// }"+
        "//@gCudaKernel<\n\n" +
        "//@>\n\n" +
        "// initialise the host data\n\n" +
        "// add the kernel code kName to end of following line\n" +
        "//@gCudaLaunchKernel\n\n" +
        "//@gCudaEndKernel\n\n" +
        "// if required emulate the whole process\n\n" +
        "// now add any final host coding\n\n"
    )

    sw.flush()
    sw.close()
  } //main

}// CreateDriverScript
