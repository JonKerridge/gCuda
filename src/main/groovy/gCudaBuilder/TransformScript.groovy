package gCudaBuilder

class TransformScript {



  static void main(String[] args) {

    def appendSemicolon = {String inLine ->
      String l = inLine
      if (l == '')return l
      if (l.trim().startsWith('//')) return l
      if (l.trim().startsWith('def')) return l
      if (l.contains('//')) {
        int p = l.indexOf('//') - 1
        while (l[p] == ' ') p = p-1
        if (l[p] =='}')
          return l
        else {
          String p1 = l.substring(0, p + 1)
          String p2 = l.substring(p + 1)
          return p1 + ';' + p2
        }
      }
      if ((l.endsWith('{')) | (l.endsWith('}'))) return l
      return l + ';'
    }

    // get all the required file names
    // it is assumed that all file will go in the same folder
    // file extensions are not required
    String gcFolderPath, gcScriptName, gcScriptFullPath, groovyOutFile, groovyFullPath
    String cKernelFileName, cKernelFullPath, ptxFileName, ptxFullPath, GPUcc

    print "Please enter directory path containing annotated script :"
//    gcFolderPath = System.in.newReader().readLine()
    gcFolderPath = "src/test/groovy/gCudaScripts"
//    gcScriptName = System.in.newReader().readLine()
    gcScriptName = "MatrixMultiply"
    print "Please enter the name of the groovy script to be transformed :"
    gcScriptFullPath = gcFolderPath + '/' + gcScriptName + '.groovy'

    print "Please enter name of resulting groovy file (also class name) :"
//    groovyOutFile = System.in.newReader().readLine()
    groovyOutFile = "gMatrixMultiply"
    groovyFullPath =gcFolderPath + '/' + groovyOutFile + '.groovy'
    print "Please enter filename of resulting C kernel file of type cu :"
//    cKernelFileName = System.in.newReader().readLine()
    cKernelFileName = "cMatrixMultiply"
    cKernelFullPath = gcFolderPath + '/' + cKernelFileName + '.cu'
    print "Please enter filename of resulting Cuda file of type ptx :"
//    ptxFileName = System.in.newReader().readLine()
    ptxFileName = "nMatrixMultiply"
    ptxFullPath = gcFolderPath + '/' + ptxFileName + '.ptx'
    println"What is the compute capability of the GPU being used\n" +
        "It takes the form of two digits xy where x.y is the GPU compute capability:"
//    GPUcc = System.in.newReader().readLine()
    GPUcc = '61'


    //read in the source script and put sections into separate structures
    List<String> imports, kernelIdiomParams, dataIn, dataOut, dataInOut, kernelParams
    List<String> groovyKernelDefinition, cKernelDefinition,initCode,  finalCode, emulation, comments, cCoding
    String kernelMethodName = ''
    (imports, kernelIdiomParams, dataIn, dataOut, dataInOut, kernelParams, groovyKernelDefinition, cKernelDefinition, initCode, emulation, finalCode, comments, cCoding) =
        [[], [], [], [], [], [], [], [], [], [], [], [], []]
    List<String> initDeviceData, getResults
    (initDeviceData, getResults) = [[], []]
    imports << 'package gCudaScripts'
    int section
    section = 0
    new File(gcScriptFullPath).eachLine {String line ->
//      println("section = $section input line is $line")
      switch (section){
        case 0:       // get the preamble
          if (!line.startsWith('//@gC'))
            imports << line
          else{
            assert line == '//@gCudaDriverKernel'
            section = 1
          }
          break
        case 1:       // kernel idiom parameters
          if (!line.startsWith('//@gC'))
            kernelIdiomParams << line
          else {
            assert line == '//@gCudaDataIn'
            section = 2
          }
          break
        case 2:       // data in variables input to GPU
          if (!line.startsWith('//@gC'))
            dataIn << line
          else {
            assert line == '//@gCudaDataOut'
            section = 3
          }
          break
        case 3:       // data out variables  output from GPU after kernel operation
          if (!line.startsWith('//@gC'))
            dataOut << line
          else {
            assert line == '//@gCudaDataInOut'
            section = 4
          }
          break
        case 4:       // bi-directional variables
          if (!line.startsWith('//@gC'))
            dataInOut << line  //also reads comments prior to @gCusaParams
          else {
            assert line.startsWith('//@gCudaKernelParams')
            kernelParams = line.tokenize(',')
            kernelParams[0] = kernelParams[0] - '//@gCudaKernelParams '
            for (i in 0..<kernelParams.size())kernelParams[i] = kernelParams[i].trim()
            kernelParams.each(p -> p.trim())
            section = 5
          }
          break
        case 5:       // typically one or more empty lines
          if (!line.startsWith('//@gC'))
            comments << line
          else {
            assert line.startsWith('//@gCudaKernel<')
            section = 6
          }
          break
        case 6:       // the kernel method definition
          if (!line.startsWith('//@')) {
            groovyKernelDefinition << line
            cKernelDefinition << appendSemicolon(line)
          }
          else {
            assert line.startsWith('//@>')
            section = 7
          }
          break
        case 7:       // the initialising code
          if (!line.startsWith('//@gC'))
            initCode << line
          else {
            assert line.startsWith('//@gCudaLaunchKernel')
            kernelMethodName = (line.tokenize())[1]
            section = 8
          }
          break
        case 8:       // emulation code, if any
          if (!line.startsWith('//@gC'))
            emulation << line
          else {
          assert line.startsWith( '//@gCudaEndKernel')
          section = 9}
          break
        case 9:       //final code
          finalCode << line
          break
      } // end switch over the input file
    } // each line
    // dataInOut contains some unwanted comment lines at the end, so remove them
      for (i in 1..3)
        dataInOut[dataInOut[].size() - i] = ""

    // now start to add text to the existing component Lists
    // import preamble
    imports << "import java.nio.Buffer"
    imports << "import static jcuda.driver.JCudaDriver.*"
    imports << "import jcuda.*"
    imports << "import jcuda.driver.*"
    imports << "import java.io.IOException"
    // now add the main classes and
    imports << "class $groovyOutFile {"
    imports << "    static void main(String[] args) throws IOException {"
    imports << "        setExceptionsEnabled(true)"
    imports << "        cuInit(0)"
    imports << "        CUdevice device = new CUdevice()  "
    imports << "        cuDeviceGet(device, 0)  "
    imports << "        CUcontext context = new CUcontext()  "
    imports << "        cuCtxCreate(context, 0, device)  "
    imports << "        // Load the PTX file containing the kernel  "
    imports << "        CUmodule module = new CUmodule()  "
    imports << "        cuModuleLoad(module, '${ptxFullPath}')  "
    imports << "        CUfunction function = new CUfunction()  "
    imports << "        cuModuleGetFunction(function, module, '${kernelMethodName}')"

// add groovy kernel definition in an appropriate place in the generated code
    for ( i in 0 ..< groovyKernelDefinition.size()) initDeviceData << groovyKernelDefinition[i]

// now create the device data allocations
//    create a map of property name as key and
    // a value of [id, direction, type]
    Map <String, List> metaData = [:]
    initDeviceData << "        final int sizeFloat = 4"
    initDeviceData << "        final int sizeDouble = 8"
    initDeviceData << "        final int intSize = 4"
    int inputID, outputID, inOutID
    (inputID, outputID, inOutID) = [0, 0, 0]
    //TODO extend this for all possible sensible data types and create a closure to do it rather than inline
    // data inputs
    for (l in 0 ..< dataIn.size()){
      List<String> dataTokens = dataIn[l].tokenize()  // assume no space in List<type>
      if (dataTokens != []) {
        String propertyName = dataTokens[1]
        if (dataTokens[0].contains('<Float>')) {
          initDeviceData << "        CUdeviceptr deviceInput$l = new CUdeviceptr();"
          initDeviceData << "        cuMemAlloc (deviceInput$l, ${propertyName}.size() * sizeFloat);"
          initDeviceData << "        cuMemcpyHtoD(deviceInput$l, Pointer.to($propertyName as float[]), ${propertyName}.size() * sizeFloat);"
          metaData.put(propertyName, [l, "Input", "List"])
        } // float
        if (dataTokens[0].contains('int')) {
          initDeviceData << "        def deviceInput$l = new int[]{$propertyName}"
          metaData.put(propertyName, [l, "Input", "Var"])
        } // int
      }
    } //for - dataIn
    //data outputs
    for (l in 0 ..< dataOut.size()){
      List<String> dataTokens = dataOut[l].tokenize()  // assume no space in List<type>
      if (dataTokens != []) {
        String propertyName = dataTokens[1]
        if (dataTokens[0].contains('<Float>')) {
          initDeviceData << "        CUdeviceptr deviceOutput$l = new CUdeviceptr();"
          initDeviceData << "        cuMemAlloc (deviceOutput$l, ${propertyName}.size() * sizeFloat);"
          metaData.put(propertyName, [l, "Output", "List"])
        } // float

      }
    } //for - dataOut
    // in out data


    // now process the metaData map in conjunction with the list of kernelParams
    // to create the required kernelParameters data structure

    println "kernelParams = ${kernelParams}"
    println "metaData = ${metaData}"
    String comma = ','
    initDeviceData << "    Pointer kernelParameters = Pointer.to("

    for ( i in 0 ..< kernelParams.size()){
      String paramName = kernelParams[i]
      List mdValue = metaData.get(paramName)
      initDeviceData << "        Pointer.to(device${mdValue[1]}${mdValue[0]})$comma "
      if ( i == kernelParams.size()-2) comma = ''
    }

    initDeviceData << "    );\n"
    initDeviceData << 'println ("kernel parameters created");\n'

    // now launch the kernel

    initDeviceData << "    cuLaunchKernel ( function,"
    initDeviceData << "        blockIdx.x, blockIdx.y, blockIdx.z,"
    initDeviceData << "        blockDim.x, blockDim.y, blockDim.z,"
    initDeviceData << "        sharedMemoryBytes, hStream as CUstream,"
    initDeviceData << "        kernelParameters, extra as Pointer );\n"
    initDeviceData << "    cuCtxSynchronize();\n"
    initDeviceData << 'println ("kernel has synchronised");\n'
    // now copy any data from GPU to host
    //TODO extend for all other output types
    for ( l in 0 ..< dataOut.size()){
      List<String> dataOutTokens = dataOut[l].tokenize()
      if (dataOutTokens != []) {
        String paramName = dataOutTokens[1]
        List metaValue = metaData.get(paramName)
        if (metaValue[2] == "List")
          if (dataOutTokens[0].contains("<Float>"))
            initDeviceData << "    cuMemcpyDtoH(Pointer.to( $paramName as float[]), deviceOutput${metaValue[0]}, ${paramName}.size() * sizeFloat );\n "
      }
    }

//    TODO also need to do for INOUT parameters


    // now clean up device allocations by adding to finalCode
    def mapValues = metaData.values()
    for ( m in 0 ..< mapValues.size())
      if (mapValues[m] == "List")
        finalCode  << "    cuMemFree(device${mapValues[m][1]}${mapValues[m][0]});"


    // now add the closing } to the final code
    finalCode << "    } //main"
    finalCode << "} //class"

    // convert the groovy kernel code to C
    cCoding << 'extern "C"'
    String defLine = "__global__ void $kernelMethodName ("
    List<String> defLineTokens = cKernelDefinition[0].tokenize()
    int pDLT = 0
    while (!defLineTokens[pDLT].startsWith('{')) pDLT = pDLT+1
    while (!defLineTokens[pDLT].endsWith('->')){
      if (defLineTokens[pDLT].contains('Float')){
        defLine = defLine + ' float '
        defLine = defLine + '*' +defLineTokens[pDLT+1].trim()
      }
      if (defLineTokens[pDLT].contains('int')){
        defLine = defLine + ' int '
        defLine = defLine + defLineTokens[pDLT+1]
      }
      pDLT = pDLT + 2
    }// while
    defLine = defLine + ') {'
    cCoding << defLine
    for ( i in 1..< cKernelDefinition.size()) cCoding << cKernelDefinition[i]

    // now test the coding above
//    println "imports"
//    imports.each(  x -> println "$x")
//    println "\nidiom params"
//    kernelIdiomParams.each(x -> println "$x")
//    println "\ndata in"
//    dataIn.each(x -> println "$x")
//    println "\ndata out"
//    dataOut.each(x -> println "$x")
//    println "\ndata in out"
//    dataInOut.each(x -> println "$x")
//    println "\nkernel params"
//    kernelParams.each(x -> println "$x")
//    println "\nkernel definition"
//    cKernelDefinition.each(x -> println "$x")
//    println "\ninitcode"
//    initCode.each(x -> println "$x")
//    println "\n device data initialise"
//    initDeviceData.each{x -> println"$x"}
//    println "kernel Method is called $kernelMethodName"
//    println "\nemulation"
//    emulation.each(x -> println "$x")
//    println "\nfinal code"
//    finalCode.each(x -> println "$x")
//    println "\n C coding"
//    cCoding.each(x -> println "$x")
    // now create the output files
    // the output Groovy file
    File gFile = new File( groovyFullPath)
    PrintWriter gFileWriter = gFile.newPrintWriter()
    // the now expanded imports
    imports.each{ line ->
      gFileWriter.println "$line"
    }
    // now data declarations including the Dim3 declaration
    dataIn.each {line ->
      gFileWriter.println "        $line"
    }
    dataOut.each {line ->
      gFileWriter.println "        $line"
    }
    dataInOut.each {line ->
      gFileWriter.println "        $line"
    }
    kernelIdiomParams.each {line ->
      gFileWriter.println "$line"
    }
    // now the data init code
    initCode.each { line ->
      gFileWriter.println "$line"
    }
    // now add the code which allocates host data and copy to the GPU
    initDeviceData.each {line ->
      gFileWriter.println "$line"
    }

    //now add any emulation coding
    emulation.each {line ->
      gFileWriter.println "$line"
    }



    finalCode.each{ line ->
      gFileWriter.println "$line"
    }
    gFileWriter.flush()
    gFileWriter.close()
    // The C code file
    File cFile = new File(cKernelFullPath)
    PrintWriter cFileWriter = cFile.newPrintWriter()
    cCoding.each{ line ->
      cFileWriter.println "$line"
    }
    cFileWriter.flush()
    cFileWriter.close()
    // now compile the C code
    String commandString = "nvcc -ptx -arch=compute_${GPUcc} ${cKernelFullPath} -o ${ptxFullPath}"
    println "Executing: $commandString"
    StringBuilder sout = new StringBuilder()
    StringBuilder serr = new StringBuilder()
    Process proc = commandString.execute()
    proc.consumeProcessOutput(sout, serr)
    proc.waitForOrKill(10000)
    println "out> $sout\nerr> $serr"

  } //main

} // class
