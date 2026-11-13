package BuilderRun

import gCuda.Build

class BuildScript {
  static void main(String[] args) {
    Build buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV1')
    buildSystem.buildApp()
  }
}
