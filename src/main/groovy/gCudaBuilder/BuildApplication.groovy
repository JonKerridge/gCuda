package gCudaBuilder

import jcuda.Sizeof

class BuildApplication {
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

    String gcFolderPath, gcScriptName, gcScriptFullPath, ptxFullPath, GPUcc
    String kernelMethodName, cFileName
    Map closureTypes = [:]
    //TODO users should be able change this default path
    gcFolderPath = "src/test/groovy/gCudaScripts"
    print "Please enter the name of the groovy script to be transformed :"
    gcScriptName = System.in.newReader().readLine()
    gcScriptFullPath = "$gcFolderPath/${gcScriptName}.groovy"

//    println"What is the compute capability of the GPU being used\n" +
//        "It takes the form of two digits xy where x.y is the GPU compute capability:"
//    GPUcc = System.in.newReader().readLine()

    File gFile, cFile
    PrintWriter gFileWriter, cFileWriter
    Map <String, List> metaData = [:]
    int propertyId = 0
    List <String> lineTokens = []
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
          metaData.put(propertyName, [propertyId, direction, propertyType - '[]', 'array'])
        else
          metaData.put(propertyName, [propertyId, direction, propertyType, 'var'])
        propertyId = propertyId + 1
      }
    }

    def getSize = { String pType ->
      String pSize
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
    }
    String cMethodDef = "__global__ void "
    boolean defProcessed = false
    // now open the script file and start to process it
    int section = 0 //control switching in following loop
    new File(gcScriptFullPath).eachLine {String line ->
      switch (section) {
        case 0:   // read and ignore import packages lines
          if (line.startsWith("//@gc")){
            assert line.startsWith('//@gcDriverKernel')
            lineTokens = line.tokenize()
            assert lineTokens.size() == 4
            // token[1] contains package details
            List<String> folderLines = lineTokens[1].tokenize('/')
            String packageName = folderLines[folderLines.size() - 1]
            String className = lineTokens[2]
            GPUcc = lineTokens[3]
//            println "$line: package: $packageName, class: $className"
            String gFileName = "${lineTokens[1]}/${className}.groovy"
            println "Groovy file is $gFileName"
            gFile = new File(gFileName)
            gFileWriter = new PrintWriter(gFile)
            cFileName = "${lineTokens[1]}/c${className}.cu"
            cFile = new File(cFileName)
            cFileWriter = new PrintWriter(cFile)
            cFileWriter.println 'extern "C"'
            ptxFullPath = "${lineTokens[1]}/n${className}.ptx"
            gFileWriter.println "package $packageName\n"
            gFileWriter.println("import gCuda.Dim3")
            gFileWriter.println("import jcuda.Pointer")
            gFileWriter.println("import jcuda.Sizeof")
            gFileWriter.println("import jcuda.driver.*")
            gFileWriter.println("import static jcuda.driver.JCudaDriver.*\n")
            gFileWriter.println("class ${className} {")
            gFileWriter.println("\tstatic void main(String[] args) {")
            section = 1
          }
          break
        case 1: //copy standard CUDA idiom declarations
          if ( !line.startsWith('//@gc'))
            gFileWriter.println "$line"
          else {
            assert  line == '//@gcKernelDefinition'
//            gFileWriter.println("\t\tString ptxFileName = '$ptxFullPath'\n")
            gFileWriter.println "$line"
            section = 2
          }
          break
        case 2: //process kernel definition closure
          if ( !line.startsWith('//@gc')) {
            gFileWriter.println "$line"
            // now process the def line
            if (!defProcessed) {
              List<String> defTokens = line.tokenize(',')
//              println "DEF: ${defTokens}"
              if (defTokens[0].trim().startsWith('def') ){
                defProcessed = true
                List<String> nameTokens = defTokens[0].tokenize()
                kernelMethodName = nameTokens[1]
                if (kernelMethodName.endsWith('=')) kernelMethodName = kernelMethodName - '='
                if (kernelMethodName.endsWith('={')) kernelMethodName = kernelMethodName - '={'
//                println "kernel method is called: $kernelMethodName"
                cMethodDef = cMethodDef + "$kernelMethodName ("
                for (i in 0..< defTokens.size()){
                  String paramName, paramType
                  boolean arrayType
                  // extract the type of all the parameters
                  List<String> paramString = defTokens[i].tokenize()
                  if (i < (defTokens.size()-1)) {
                     paramName = paramString[paramString.size() - 1]
                     paramType = paramString[paramString.size() - 2]  // could be var or array
                  }
                  else {  //last param has to ignore ->
                    if (paramString[paramString.size() - 1].trim() == '->'){
                      paramName = paramString[paramString.size() - 2]
                      paramType = paramString[paramString.size() - 3]  // could be var or array
                    }
                    else{
                      paramName = paramString[paramString.size() - 1]
                      paramType = paramString[paramString.size() - 2]  // could be var or array
                    }
                  }
                  if (paramType.contains('int') ) cMethodDef = cMethodDef + 'int '
                  if (paramType.contains('float') ) cMethodDef = cMethodDef + 'float '
                  if (paramType.contains('double') ) cMethodDef = cMethodDef + 'double '
                  if (paramType.contains('[]')) cMethodDef = cMethodDef + "*$paramName"
                  else cMethodDef = cMethodDef + "$paramName"
                  if ( i < (defTokens.size()-1)) cMethodDef = cMethodDef + ','
                } // defTokens loop
                cMethodDef = cMethodDef + ") {"
                cFileWriter.println("$cMethodDef")
              } // starts with def
              else {  //def line, expected but not found
                println "$line :: read, but expected a closure definition line"
              }
            }  // def not processed
            else{ //  def has been processed and reading rest of closure
              // just have to append ; to each of the lines
              cFileWriter.println("${appendSemicolon(line)}")
            }
          } // line starts with //@gc
          else {
            assert line == '//@gcDataToGPU'
            gFileWriter.println "$line"
            section = 3
          }
          break
        case 3: //process data to GPU
          if (!line.startsWith('//@gc')){
            // process properties transferred to GPU from host
            gFileWriter.println "$line"
            extractDataTypes(line, 'ToGPU')
          }
          else {
            assert line == '//@gcDataFromGPU'
            gFileWriter.println "$line"
            section = 4
          }
          break
        case 4: //process data from GPU
          if (!line.startsWith('//@gc')){
            // process properties transferred from GPU to host
            gFileWriter.println "$line"
            extractDataTypes(line, 'FromGPU')
          }
          else {
            assert line == '//@gcDataBoth'
            gFileWriter.println "$line"
            section = 5
          }
          break
        case 5: //process data passed bidirectionally between host and GPU
          if (!line.startsWith('//@gc')){
            // process properties transferred in both directions
            gFileWriter.println "$line"
            extractDataTypes(line, 'Both')
          }
          else {
            assert line == '//@gcDataInitialise'
            gFileWriter.println "$line"
            section = 6
          }
          break
        case 6: //process data initialisation and then kernel parameters
          if (!line.startsWith('//@gc')){
            // process initialisation code lines
            gFileWriter.println "$line"
          }
          else {
            assert line.startsWith('//@gcKernelParams')
            gFileWriter.println "$line"
//            println "$line"

            // process the kernel parameters annotation
            lineTokens = line.tokenize(',')
            lineTokens[0] = lineTokens[0] - '//@gcKernelParams '
//            println "Parameters are: ${lineTokens}"
            // now generate most of the required code
            // device context, module and function initialisation
            gFileWriter.println "\t\tsetExceptionsEnabled(true)"
            gFileWriter.println "\t\tcuInit(0)"
            gFileWriter.println "\t\tCUdevice device = new CUdevice() "
            gFileWriter.println "\t\tcuDeviceGet(device, 0) "
            gFileWriter.println "\t\tCUcontext context = new CUcontext()"
            gFileWriter.println "\t\tcuCtxCreate(context, 0, device)"
            gFileWriter.println "\t\t// Load the PTX file containing the kernel "
            gFileWriter.println "\t\tCUmodule module = new CUmodule() "
            gFileWriter.println "\t\tcuModuleLoad(module, '${ptxFullPath}') "
            gFileWriter.println "\t\tCUfunction function = new CUfunction()"
            gFileWriter.println "\t\tcuModuleGetFunction(function, module, '${kernelMethodName}')"
//            println "metaData = ${metaData}"
            // now allocate the memory for each of the parameters
            // and copy data as necessary
            int nParams = lineTokens.size()
//            println "line tokens is $lineTokens"
            for ( i in 0 ..< nParams){
              String paramName = lineTokens[i].trim()
//              println "Section 6 : ${metaData}"
              List metaInf = metaData.get(paramName)
//              println "PName = $paramName,  metainf = ${metaInf}"
              int paramId = (int)metaInf[0]
              String paramDir = metaInf[1]
              String paramType = metaInf[2]
              if (metaInf[3] == 'var') {
                // dealing with a variable
                gFileWriter.println "\t\t${paramType}[] param$paramDir$paramId = new ${paramType}[]{$paramName} "
                // associate internal parameter name with text name
//                println "\t\t${paramType}[] param$paramDir$paramId = new ${paramType}[]{$paramName} "
//                paramSize = getSize(paramType)
              }
              else{
                // dealing with an array
                String paramSize = getSize(paramType)
                gFileWriter.println "\t\tCUdeviceptr param$paramDir$paramId = new CUdeviceptr()"
                gFileWriter.println "\t\tcuMemAlloc (param$paramDir$paramId, ${paramName}.size() * $paramSize)"
                if ((paramDir == 'ToGPU') || (paramDir == 'Both') ) // then copy values to GPU for all values sent to GPU
                  gFileWriter.println "\t\tcuMemcpyHtoD (param$paramDir$paramId, Pointer.to($paramName), ${paramName}.size() * $paramSize)"
              }
            } // for line tokens
            // now set up the kernel parameters structure
            gFileWriter.println "\t\tPointer kernelParameters = Pointer.to ("
            for ( i in 0 ..< nParams) {
              String comma = ','
              if (i == (nParams-1)) comma = ' '
              List mapValues = metaData.get(lineTokens[i].trim())
//              println "For param ${lineTokens[i]}, MapValues are ${mapValues}"
              gFileWriter.println "\t\t\tPointer.to( param${mapValues[1]}${mapValues[0]})$comma"
            }
            gFileWriter.println "\t\t)"
            // now call the kernel function
            gFileWriter.println "\t\tcuLaunchKernel ( function,"
            gFileWriter.println "\t\t\tgridSize.x, gridSize.y, gridSize.z,"
            gFileWriter.println "\t\t\tblockSize.x, blockSize.y, blockSize.z,"
            gFileWriter.println "\t\t\tsharedMemoryBytes, hStream,"
            gFileWriter.println "\t\t\tkernelParameters, extra"
            gFileWriter.println "\t\t\t)"
            gFileWriter.println "\t\tcuCtxSynchronize()"
            // now copy any data from GPU to host as required
            for ( i in 0 ..< nParams){
              String paramName = lineTokens[i].trim()
              List metaInf = metaData.get(paramName)
              if ( (metaInf[1] == 'FromGPU') || (metaInf[1] == 'Both')){
                gFileWriter.println "\t\tcuMemcpyDtoH(Pointer.to(${paramName}), " +
                    "param${metaInf[1]}${metaInf[0]}, ${paramName}.size() * ${getSize((String)metaInf[2])})"
              }
            }
            section = 7
          }
          break
        case 7: // absorb kernel end
          if (!line.startsWith('//@gc')){
            gFileWriter.println "$line"
          }
          else {
            assert line == '//@gcKernelEnd'
            gFileWriter.println "$line"
            // generate code to copy outputs fromGPU to host
            section = 8
          }

          break
        case 8: // copy any lines as is
          if (!line.startsWith('//@gc')){
            gFileWriter.println "$line"
          }
          else {
            assert line == '//@gcEmulate'
            gFileWriter.println "$line"
            section = 9
          }
          break
        case 9: //copy any lines as is
          if (!line.startsWith('//@gc')){
            gFileWriter.println "$line"
          }
          else {
            assert line == '//@gcFinalise'
            gFileWriter.println "$line"
            section = 10
          }
          break
        case 10:
          if (!line.startsWith('//@gc')){
            gFileWriter.println "$line"
          }
          else {
            assert line == '//@gcFinish'
            gFileWriter.println "$line"
            // generate code required to clean up array memory
            def metaValues= metaData.values()
            for ( i in 0 ..< metaData.size()){
              if (metaValues[i][3] == 'array')
                gFileWriter.println "\t\tcuMemFree(param${metaValues[i][1]}${metaValues[i][0]})"
            }
            section = 11
          }

          break
        case 11: // copy any remaining lines as is
          gFileWriter.println "$line"
          break
      } // end switch
    }// end closure loop





    gFileWriter.println "  }"
    gFileWriter.println "}"
    // close both files
    gFileWriter.flush()
    gFileWriter.close()
    cFileWriter.flush()
    cFileWriter.close()
    // now compile the C code into a ptx file
    String commandString = "nvcc -ptx -arch=compute_${GPUcc} ${cFileName} -o ${ptxFullPath}"
    println "Executing: $commandString"
    StringBuilder sOut = new StringBuilder()
    StringBuilder sErr = new StringBuilder()
    Process proc = commandString.execute()
    proc.consumeProcessOutput(sOut, sErr)
    proc.waitForOrKill(15000)
    println "out> $sOut\nerr> $sErr"
  } // main
} // class

