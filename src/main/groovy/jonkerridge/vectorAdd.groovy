package jonkerridge
// required global declarations to access vectors
int  blockDim_x, blockIdx_x, threadIdx_x

//@gCudaDataIn
List x,y
int vSize

//@gCudaDataOut
List z

//@gCudaDataInOut



//@gCudaKernel<
add = {int n, List a, List b, List c ->
  int i = blockIdx_x * blockDim_x + threadIdx_x
  if (i < n){
    c[i] = a[i] + b[i]
  }
}
//@>

// initialise the host data
vSize = 9001
x = []
y = []
z = []
for ( i in 0 ..< vSize){
  x[i] = i
  y[i] = i
  z[i] = i
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
    add(vSize, x, y, z)
  }
}
//@gCudaEndKernel

for ( i in 0 ..< vSize) {
  assert x[i] == i
  assert y[i] == i
  assert z[i] == 2 * i
}

println "Finished"
for ( i in 0 ..< vSize) print "${z[i]}, "
println ""



