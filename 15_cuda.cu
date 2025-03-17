__global__ void add (int* a, int* b, int* c){
    int tid = threadIDx.x + blockIdx.x * blockDimx.x;
    
}