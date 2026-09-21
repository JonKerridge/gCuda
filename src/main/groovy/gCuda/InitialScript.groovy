package gCuda

class InitialScript {
    String gCudaScriptName, gCudaScriptPath

    InitialScript(){
        // this version requires the user to input
        // package name and filename
        print "Please enter the path for the groovyCuda script to generate:"
        gCudaScriptPath = System.in.newReader().readLine()
        print "Please enter the name of the groovyCuda script to generate:"
        gCudaScriptName = System.in.newReader().readLine()
        createScript(gCudaScriptPath, gCudaScriptName)
    }

  /**
   *
   * @param gCudaSP the path to the directory holding the script file
   * @param gCudaSN the name of the script file
   */
    InitialScript(String gCudaSP,
                  String gCudaSN ){
        gCudaScriptPath = gCudaSP
        gCudaScriptName = gCudaSN
        createScript(gCudaScriptPath, gCudaScriptName)
    }
/**
 *
 * @param path the gCudaScriptPath
 * @param name the gCudaScriptName, assumed to be nameScript.groovy
 */
  static createScript (String path, String name){
        String scriptName
        scriptName = path + name + "Script.groovy"
        File scriptFile = new File (scriptName)
        if (scriptFile.exists()) scriptFile.delete()
        PrintWriter sw = scriptFile.newPrintWriter()

        sw.println("import gCuda.Dim3")
        sw.println("import jcuda.Pointer")
        sw.println("import jcuda.Sizeof")
        sw.println("import jcuda.driver.*")
        sw.println("import static jcuda.driver.JCudaDriver.*\n")

        sw.println ("//@gcDriverKernel   path  fileName GPUcompute\n\n" +
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
                        "blockDim = blockSize\n\n" +
                        "//@gcFinalise\n\n" +
                        "//@gcFinish\n\n"
        )
        sw.flush()
        sw.close()

    } // createScript

    static void main(String[] args) {
        String gCudaScriptName, gCudaScriptPath
        if (args == []){
            print "Please enter the path for the groovyCuda script to generate:"
            gCudaScriptPath = System.in.newReader().readLine()
            print "Please enter the name of the groovyCuda script to generate:"
            gCudaScriptName = System.in.newReader().readLine()

        }
        else{
            gCudaScriptPath = args[0]
            gCudaScriptName = args[1]
        }
        if (!gCudaScriptPath.trim().endsWith('/'))
          gCudaScriptPath = gCudaScriptPath + '/'
        createScript(gCudaScriptPath, gCudaScriptName)
    }
}
