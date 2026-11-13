package gCuda

class Build {
  String gcScriptName, gcScriptPath

  Build() {
    // this version requires the user to input
    // package name and filename
    print "Please enter the path for the groovyCuda script to build:"
    gcScriptPath = System.in.newReader().readLine()
    print "Please enter the name of the groovyCuda script to build (omit Script.groovy suffix):"
    gcScriptName = System.in.newReader().readLine()
  }

  /**
   *
   * @param gcScriptPath the possibly relative, to the Working Directory of the script file
   * @param gcScriptName the name of the script file without the Script.groovy suffix
   */
  Build(String gcSP,
          String gcSN){
    gcScriptPath =  gcSP
    gcScriptName = gcSN
  }

  // parsing data structures
  Map <String, List> metaData = [:]
  //Map with propertyName as key and a list as value comprising
  // [ propertyId, direction (To, From, Both), type, array or var, declared]
  // declared will be true once an instance of the property has already been declared
  // and is now being used in a second or subsequent call
  int propertyId = 0
  List <String> lineTokens
  List<String> script = []
  List<Integer> keyLines = []
  String cMethodDef = "__global__ void "
  boolean defProcessed = false
  boolean processingC = false
  boolean postCprocessing = false
  List<String> kernelMethodNames = []

  // files etc
  String gcScriptFullPath, ptxFullPath, GPUcc
  String gFileName, cFileName, kernelNature, packageName
  File gFile, cFile
  PrintWriter gFileWriter, cFileWriter



  // support closures
  def appendSemicolon = {String inLine ->
    // used to convert the kernel closure def to C
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
  } // append semicolon

  def extractDataTypes = {String line, String direction ->
    // each line contains a type declaration, property name and
    // an initialisation; it assumes any = sign is surrounded by spaces
    lineTokens = line.tokenize()
    //deal with a possible blank line
    if (lineTokens != []) {
      assert lineTokens[2].trim() == '='
      String propertyType = lineTokens[0].trim()
      String propertyName = lineTokens[1].trim()
      if (propertyType.contains('[]'))
        metaData.put(propertyName, [propertyId, direction, propertyType - '[]', 'array', false])
      else
        metaData.put(propertyName, [propertyId, direction, propertyType, 'var', false])
      propertyId = propertyId + 1
    }
  } // extract data types

  def getSize = { String pType ->
    String pSize = ''
    switch (pType) {
      case 'int':
        pSize = 'Sizeof.INT'
        break
      case 'float':
        pSize = 'Sizeof.FLOAT'
        break
      case 'double':
        pSize = 'Sizeof.DOUBLE'
        break
      case 'byte':
        pSize = 'Sizeof.BYTE'
        break
      case 'long':
        pSize = 'Sizeof.LONG'
        break
      case 'pointer':
        pSize = 'Sizeof.POINTER'
        break
      case 'short':
        pSize = 'Sizeof.SHORT'
        break
      case 'char':
        pSize = 'Sizeof.CHAR'
        break
    }
    return pSize
  } // get size

/**
 * @param start index of first script line  defining the kernel
 * @param last the index of the line containing the next //@gc
 */

  void processGroovyDef(int start, last) {
    int current
    current = start
    String line
    while (current < last) {
      line = script[current]
      gFileWriter.println "\t$line"
      // now process the def line
      if (!defProcessed) {
        List<String> defTokens = line.tokenize(',')
//              println "DEF: ${defTokens}"
        if (defTokens[0].trim().startsWith('def')) {
          defProcessed = true
          List<String> nameTokens = defTokens[0].tokenize()
          String kernelMethodName = nameTokens[1]
          kernelMethodNames << kernelMethodName
          if (kernelMethodName.endsWith('=')) kernelMethodName = kernelMethodName - '='
          if (kernelMethodName.endsWith('={')) kernelMethodName = kernelMethodName - '={'
//                println "kernel method is called: $kernelMethodName"
          cMethodDef = cMethodDef + "$kernelMethodName ("
          for (i in 0..<defTokens.size()) {
            String paramName, paramType
            boolean arrayType
            // extract the type of all the parameters
            List<String> paramString = defTokens[i].tokenize()
            if (i < (defTokens.size() - 1)) {
              paramName = paramString[paramString.size() - 1]
              paramType = paramString[paramString.size() - 2]
              // could be var or array
            } else {  //last param has to ignore ->
              if (paramString[paramString.size() - 1].trim() == '->') {
                paramName = paramString[paramString.size() - 2]
                paramType = paramString[paramString.size() - 3]
                // could be var or array
              } else {
                paramName = paramString[paramString.size() - 1]
                paramType = paramString[paramString.size() - 2]
                // could be var or array
              }
            }
            if (paramType.contains('int')) cMethodDef = cMethodDef + 'int '
            if (paramType.contains('float')) cMethodDef = cMethodDef + 'float '
            if (paramType.contains('double')) cMethodDef = cMethodDef + 'double '
            if (paramType.contains('[]')) cMethodDef = cMethodDef + "*$paramName" else cMethodDef = cMethodDef + "$paramName"
            if (i < (defTokens.size() - 1)) cMethodDef = cMethodDef + ','
          } // defTokens loop
          cFileWriter.println 'extern "C"'
          cMethodDef = cMethodDef + ") {"
          cFileWriter.println("$cMethodDef")
        } // starts with def
        else {  //def line, expected but not found
          println "$line :: read, but expected a closure definition line"
        }
      }  // def not processed
      else { //  def has been processed and reading rest of closure
        // just have to append ; to each of the lines
        cFileWriter.println("${appendSemicolon(line)}")
      }
      current = current + 1
    } // while loop
    cFileWriter.flush()
    cFileWriter.close()
  } // processGroovyDef

  /**
   * @param start index of first script line  defining the kernel
   * @param last the index of the line containing the next //@gc
  */

  void processCDef(int start, last){
    int current
    current = start
    String line
    while (current < last) {
      line = script[current]
      if (!postCprocessing) {
        if ((line != '/*@gc') && (!processingC)) {
          // blank or other comment lines
          gFileWriter.println "\t$line"
        } else {
          // now process the /*@gc line
          if ((line == '/*@gc') && (!processingC)) {
            gFileWriter.println "\t/*@gc start of C code"
            processingC = true
//                                println "$line and processingC = $processingC"
          } else {  //now found the C statements
            if ((line != '*/') && (processingC)) {
              // extract kernel method name from --global-- line
              if (line.contains('__global__')) {
                int c = 16
                while ((line[c] != ' ') && (line[c] != '(')) c = c + 1
                kernelMethodNames << line.substring(16, c).trim()
              }
              gFileWriter.println "\t$line"
              // copy line directly to groovy output
              cFileWriter.println "$line" // copy line to cu file
            } else {
              // found the */ end of C comment line
              gFileWriter.println "\tend of C coding $line"
              // copy to groovy output
              postCprocessing = true
              cFileWriter.flush()
              cFileWriter.close()
            }
          }
        }
      }
      else // copy any blank and comment lines after the C code
        gFileWriter.println "$line"
      current = current + 1
    } // while loop

  } // processCDef

  /**
   * @param start index of first script line  defining the data definitions
   * @param last the index of the line containing the next //@gc
   */
  void extractDataToGPU(int start, last){
    for ( i in start ..< last){
      // process properties transferred to GPU from host
      gFileWriter.println "\t${script[i]}"
      extractDataTypes(script[i], 'ToGPU')
    }
  }

  /**
   * @param start index of first script line  defining the data definitions
   * @param last the index of the line containing the next //@gc
   */
  void extractDataFromGPU(int start, last){
    for ( i in start ..< last){
      // process properties transferred to GPU from host
      gFileWriter.println "\t${script[i]}"
      extractDataTypes(script[i], 'FromGPU')
    }
  }

  /**
   * @param start index of first script line  defining the data definitions
   * @param last the index of the line containing the next //@gc
   */
  void extractDataBoth(int start, last){
    for ( i in start ..< last){
      // process properties transferred to GPU from host
      gFileWriter.println "\t${script[i]}"
      extractDataTypes(script[i], 'Both')
    }
  }

  /**
   * @param start index of first script line  first line to be copied
   * @param last the index of the line containing the next //@gc
   */
  void copyLines(int start, last){
    for ( i in start ..< last){
      // process properties transferred to GPU from host
      gFileWriter.println "\t${script[i]}"
    }
  }

  /**
   *
   * @param index the index of the //@gcCall specification
   */
  void extractCall(int index, boolean firstCall){
    String line = script[keyLines[index]]
    line = line - '//@gcCall '
    lineTokens = line.tokenize('(')
    String kernelMethodName = lineTokens[0].trim()
    assert kernelMethodNames.contains(kernelMethodName)
    line = lineTokens[1] - ')'    // the parameters with a trailing )
    if (firstCall)
      gFileWriter.println "\tCUfunction function = new CUfunction()"
    gFileWriter.println "\tcuModuleGetFunction(function, module, '${kernelMethodName}')"
    lineTokens = line.tokenize(',')
    println "$lineTokens"
    int nParams = lineTokens.size()
    for ( i in 0 ..< nParams){
      String paramName = lineTokens[i].trim()
      List metaInf = metaData.get(paramName)
      int paramId = (int)metaInf[0]
      String paramDir = metaInf[1]
      String paramType = metaInf[2]
      String arrayOrVar = metaInf[3]
      boolean declared = metaInf[4]
      if (arrayOrVar == 'var')
        if (!declared) {
        // declare the array structure for the single value and update declared in map
        // can be reused in subsequent calls
        gFileWriter.println "\t${paramType}[] param$paramDir$paramId = new ${paramType}[]{$paramName} "
        metaData.put(paramName, [paramId, paramDir, paramType, 'var', true])
        }
        else{} // do nothing
      else {  // dealing with an array
        if (!declared){
          // declare the device pointer for the array and update declared in map
          gFileWriter.println "\tCUdeviceptr param$paramDir$paramId = new CUdeviceptr()"
          metaData.put(paramName, [paramId, paramDir, paramType, 'array', true])
        }
        String paramSize = getSize(paramType)
        gFileWriter.println "\tcuMemAlloc (param$paramDir$paramId, ${paramName}.size() * $paramSize)"
        if ((paramDir == 'ToGPU') || (paramDir == 'Both') ) // then copy values to GPU for all values sent to GPU
          gFileWriter.println "\tcuMemcpyHtoD (param$paramDir$paramId, Pointer.to($paramName), ${paramName}.size() * $paramSize)"
      }
    } // for line tokens
    // now set up the kernel parameters structure
    if (firstCall)
      gFileWriter.println "\tPointer kernelParameters = new Pointer()"
    gFileWriter.println "\tkernelParameters = Pointer.to ("
    for ( i in 0 ..< nParams) {
      String comma = ','
      if (i == (nParams-1)) comma = ' '
      List mapValues = metaData.get(lineTokens[i].trim())
      gFileWriter.println "\t\tPointer.to( param${mapValues[1]}${mapValues[0]})$comma"
    }
    gFileWriter.println "\t)"
    // now call the kernel function
    gFileWriter.println "\tcuLaunchKernel ( function,"
    gFileWriter.println "\t\tgridSize.x, gridSize.y, gridSize.z,"
    gFileWriter.println "\t\tblockSize.x, blockSize.y, blockSize.z,"
    gFileWriter.println "\t\tsharedMemoryBytes, hStream,"
    gFileWriter.println "\t\tkernelParameters, extra"
    gFileWriter.println "\t\t)"
    gFileWriter.println "\tcuCtxSynchronize()"
    // now copy any data from GPU to host as required
    for ( i in 0 ..< nParams){
      String paramName = lineTokens[i].trim()
      List metaInf = metaData.get(paramName)
      if ( (metaInf[1] == 'FromGPU') || (metaInf[1] == 'Both')){
        gFileWriter.println "\tcuMemcpyDtoH(Pointer.to(${paramName}), " +
            "param${metaInf[1]}${metaInf[0]}, ${paramName}.size() * ${getSize((String)metaInf[2])})"
      }
    } // for nParams loop
  }

  /**
   * buildApp the primary method of the class used to create the
   * the required output files for use in a host CPU with GPU
   *
   * The script file can contain either a method defined as a Groovy
   * closure, provided no special Cuda calls or with in C code otherwise.
   * The required C definition of the kernel will be derived from
   * the Groovy kernel closure definition.
   *
   * If C is used the code will be used as is.
   *
   * The Builder will create a CUDA .ptx equivalent file using
   * the installed nvcc compiler from the generated .cu file
   *
   *
   * The current latest version of the NVIDIA toolset that can be
   * used is 12.6.0, which may limit the GPU cards that can be used.
   *
   */

  void buildApp (){
    if (!gcScriptPath.trim().endsWith('/'))
      gcScriptPath = gcScriptPath + '/'
    gcScriptFullPath = "$gcScriptPath${gcScriptName}Script.groovy"
    println "Building $gcScriptFullPath"

    // read in script file  to script
    new File(gcScriptFullPath).eachLine {String line ->
      script << line
    }
    // find lines starting with //@gc
    for ( i in 0..< script.size())
      if (script[i].startsWith('//@gc'))
        keyLines << i
//    for ( i in 0 ..< script.size())
//      println "$i\t${script[i]}"
//    println "keyLines = $keyLines"
    lineTokens = gcScriptPath.tokenize('/')
    packageName = lineTokens[lineTokens.size()-1]
    // initialise the output files
    gFileName = "$gcScriptPath$gcScriptName" + ".groovy"
    println "Groovy executable file is $gFileName"
    gFile = new File(gFileName)
    gFileWriter = new PrintWriter(gFile)
    cFileName = "$gcScriptPath$gcScriptName" + ".cu"
    cFile = new File(cFileName)
    cFileWriter = new PrintWriter(cFile)
    ptxFullPath = "$gcScriptPath$gcScriptName" + ".ptx"
    gFileWriter.println "package $packageName\n"
    for ( i in 0 ..< keyLines[0])
      gFileWriter.println "${script[i]}"
    assert script[keyLines[0]].startsWith('//@gcDriverKernel') :
      "On Line ${keyLines[0]} expecting //@gcDriverKernel ... found ${script[keyLines[0]]}"
    gFileWriter.println("class ${gcScriptName} {")
    gFileWriter.println("\tstatic void main(String[] args) {")
    // process the //@gcDriverKernel line
    lineTokens = script[keyLines[0]].tokenize()
    GPUcc = lineTokens[1]
//    println "GPU cc = $GPUcc"
    // now copy initial declarations before kernel definition
    for ( i in keyLines[0]+1 ..< keyLines[1])
      gFileWriter.println('\t' + script[i])
    assert script[keyLines[1]].startsWith('//@gcKernelDefinition')
    assert script[keyLines[1]].endsWith(('C')) || script[keyLines[1]].endsWith('G'):
        "kernelNature (G or C) in kernel definition annotation not specified correctly"
    kernelNature = script[keyLines[1]].substring(script[keyLines[1]].length()-1)
    gFileWriter.println('\t' + script[keyLines[1]])
    if (kernelNature == 'G')
      processGroovyDef(keyLines[1] + 1, keyLines[2])
    else
      processCDef(keyLines[1] + 1, keyLines[2])
    // now process the //@gcData annotations
    int keyIndex
    keyIndex = 2  // data annotations start at 2
    while (script[keyLines[keyIndex]] != '//@gcDataInitialise'){
      gFileWriter.println('\t' + script[keyLines[keyIndex]])
      switch (script[keyLines[keyIndex]]){
        case '//@gcDataToGPU':
          extractDataToGPU(keyLines[keyIndex] + 1, keyLines[keyIndex + 1])
          break
        case '//@gcDataFromGPU':
          extractDataFromGPU(keyLines[keyIndex] + 1, keyLines[keyIndex + 1])
          break
        case '//@gcDataBoth' :
          extractDataBoth(keyLines[keyIndex] + 1, keyLines[keyIndex + 1])
          break
      }
      keyIndex = keyIndex + 1
    }
    assert script[keyLines[keyIndex]].trim() == '//@gcDataInitialise'
    assert script[keyLines[keyIndex + 1]].trim() == '//@gcKernelInit'
    gFileWriter.println('\t' + script[keyLines[keyIndex]])
    // now copy lines directly to gFileWriter
    copyLines(keyLines[keyIndex] + 1, keyLines[keyIndex + 1])
    keyIndex = keyIndex + 1 // now index of //@gcKernelInit
    gFileWriter.println('\t' + script[keyLines[keyIndex]])
    // add the lines of JCuda initialisation code
    gFileWriter.println "\tsetExceptionsEnabled(true)"
    gFileWriter.println "\tcuInit(0)"
    gFileWriter.println "\tCUdevice device = new CUdevice() "
    gFileWriter.println "\tcuDeviceGet(device, 0) "
    gFileWriter.println "\tCUcontext context = new CUcontext()"
    gFileWriter.println "\tcuCtxCreate(context, 0, device)"
    gFileWriter.println "\t// Load the PTX file containing the kernel "
    gFileWriter.println "\tCUmodule module = new CUmodule() "
    gFileWriter.println "\tcuModuleLoad(module, '${ptxFullPath}') "
    // copy any intervening inline code between //@gcKernelInit and first //@gcCall
    copyLines(keyLines[keyIndex] + 1, keyLines[keyIndex + 1])
    assert script[keyLines[keyIndex + 1]].startsWith('//@gcCall')
    keyIndex = keyIndex + 1  // keyIndex is index of first or only //@gcCall
    extractCall(keyIndex, true)
    // now process any subsequent //@gcCall lines
    while (script[keyLines[keyIndex + 1]].startsWith('//@gcCall')){
      copyLines(keyLines[keyIndex] + 1, keyLines[keyIndex + 1])
      keyIndex = keyIndex + 1  // keyIndex is index of a //@gcCall
      extractCall(keyIndex, false)
    }
    assert script[keyLines[keyIndex + 1]].startsWith('//@gcKernelEnd')
    keyIndex = keyIndex + 1 // index of //@gcKernelEnd
    gFileWriter.println("\t${script[keyLines[keyIndex]]}")
    assert script[keyLines[keyIndex + 3]].startsWith('//@gcFinish')
    copyLines(keyLines[keyIndex] + 1, keyLines[keyIndex + 3])
    keyIndex = keyIndex + 3
    gFileWriter.println("\t${script[keyLines[keyIndex]]}")
    def metaValues= metaData.values()
    for ( i in 0 ..< metaData.size()){
      if (metaValues[i][3] == 'array')
        gFileWriter.println "\tcuMemFree(param${metaValues[i][1]}${metaValues[i][0]})"
    }
    for ( i in keyLines[keyIndex]+1 ..< script.size())
      gFileWriter.println("\t${script[i]}")

    // insert closing } for main and class
    gFileWriter.println ("\t} // main")
    gFileWriter.println ("} // $gcScriptName")
    gFileWriter.flush()
    gFileWriter.close()

    // now compile the C code into a ptx file
    String commandString = "nvcc -ptx -arch=compute_${GPUcc} ${cFileName} -o ${ptxFullPath}"
    println "Executing: $commandString"
    StringBuilder sOut = new StringBuilder()
    StringBuilder sErr = new StringBuilder()
    Process proc = commandString.execute()
    proc.consumeProcessOutput(sOut, sErr)
    proc.waitForOrKill(15000)
    println "out> $sOut\nerr> $sErr"
  }// buildApp

}//build
