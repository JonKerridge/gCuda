package jonkerridge
// required global declarations to access vectors
int  blockDim_x, blockIdx_x, threadIdx_x

//@gCudaDataIn
int vSize, scaleValue

//@gCudaDataOut


//@gCudaDataInOut
List v


//@gCudaKernel<
scale = {int n, List vec, int scale ->
  int i = blockIdx_x * blockDim_x + threadIdx_x
  if (i < n){
    vec[i] = vec[i] * scale
  }
}
//@>

// initialise the host data
vSize = 9001
scaleValue = 2
v = []

for ( i in 0 ..< vSize){
  v[i] = i

}

// now set the block and grid size
int blockSizeX = 256
int gridSizeX = (int)Math.ceil((double)vSize / blockSizeX)
println "Gridsize is $gridSizeX"

//@gCudaLaunchKernel add, blockSizeX, gridSizeX


// now emulate the whole process
blockDim_x = blockSizeX

for (bId in 0 ..< gridSizeX) {
  blockIdx_x = bId
  for (tId in 0..<blockSizeX) {
    threadIdx_x = tId
    scale(vSize, v, scaleValue)
  }
}
//@gCudaEndKernel

for ( i in 0 ..< vSize) {
  assert v[i] == scaleValue * i
}

println "Finished"
for ( i in 0 ..< vSize) print "${v[i]}, "
println ""



