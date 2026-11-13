extern "C"
__global__ void add(int n, int *a, int *b, int *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        sum[i] = a[i] + b[i];
    }

}

extern "C"
__global__ void scalerX(int n, int *a, int s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        a[i] = a[i] * s;
    }

}

