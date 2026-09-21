package gCudaScripts

import gCuda.Builder

class BuildScript {
  static void main(String[] args) {
    Builder buildScript = new Builder('D:/IJGradle/gCUDA/src/test/groovy/gCudaScripts/', 'daft20')
    buildScript.build()
  }


}
