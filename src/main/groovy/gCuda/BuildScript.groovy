package gCuda

class BuildScript {
  static void main(String[] args) {
    Build buildSystem = new Build('src/test/groovy/gCudaScripts', 'AsymMtrxMulply')
    buildSystem.buildApp()
  }

}
