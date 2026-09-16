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
      print "Please enter the name of the groovyCuda script to generate:"
      gCudaScriptPath = "src/test/groovy/gCudaScripts/"
      gCudaScriptName = System.in.newReader().readLine()
    }
    else {
      gCudaScriptPath = args[0]
      gCudaScriptName = args[1]
    }
    println "creating: $gCudaScriptPath.$gCudaScriptName"
    String scriptName = gCudaScriptPath + gCudaScriptName + ".groovy"
    File scriptFile = new File (scriptName)
    PrintWriter sw= new PrintWriter(scriptFile)

    sw.println("import gCuda.Dim3")
    sw.println("import jcuda.Pointer")
    sw.println("import jcuda.Sizeof")
    sw.println("import jcuda.driver.*")
    sw.println("import static jcuda.driver.JCudaDriver.*\n")

    sw.println ("//@gcDriverKernel   path  appName\n\n" +
        "Dim3 gridSize = new Dim3()    // must be initialised in the DataInitialise phase\n" +
        "Dim3 blockSize = new Dim3()   \n" +
        "int sharedMemoryBytes = 0\n" +
        "CUstream hStream = null \n" +
        "Pointer extra = null \n\n" +
        "// CUDA idiomatic properties\n" +
        "Dim3 blockIdx = new Dim3()\n" +
        "Dim3 blockDim = new Dim3()\n" +
        "Dim3 threadIdx = new Dim3()\n\n"
    )

    sw.println(
        "//@gcKernelDefinition\n\n" +
        "//@gcDataToGPU\n\n" +
        "//@gcDataFromGPU\n\n" +
        "//@gcDataBoth\n\n" +
        "//@gcDataInitialise\n\n" +
        "blockSize.x = 32   \n"+
        "// determine number of blocks in grid, modify following as required\n" +
        "gridSize.x = (int)Math.ceil((double) DATASIZE / blockSize.x)\n"+
        "//@gcKernelParams \n\n" +
        "//@gcKernelEnd\n\n" +
        "//@gcEmulate\n\n" +
        "//@gcFinalise\n\n" +
        "//@gcFinish\n\n"
    )
    sw.flush()
    sw.close()
  } //main

}// CreateDriverScript
