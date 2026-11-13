package BuilderRun

import gCuda.Build

class BuildAll {
  static void main(String[] args) {
    Build buildSystem = new Build('src/test/groovy/gCudaScripts', 'AsymMatrixMultiply')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'DynamicSharedMemory')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'FloatAXPY')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'FusedMultiplyAdd')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'MatrixMultiply')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'ScalerMultiply')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'StaticSharedMemory')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/gCudaScripts', 'VectorAdd')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV1')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV2')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV3')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV4')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV5')
    buildSystem.buildApp()
    buildSystem = new Build('src/test/groovy/ParallelReduction', 'ReductionV7')
    buildSystem.buildApp()
  }
}
