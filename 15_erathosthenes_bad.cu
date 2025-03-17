typedef uint32_t u32;

int prime_count = 0;


__global__
void erathostenes(u32* primes, u32 n) {
    int tid = threadIDx.x; //each thread handles one element 
    u32 count = 0; //where is this? who owns this?

   

    //here's a probably better idea:
    //let's all cooperate to write to memory[2, 1024]
    //next thread can write [1025 .... 2048]

    //each thread will check numbers tid, tid + 1, tid + 2 ....
    u32 start = tid*2+1, end = start + 64;
    
    for (u32 i = start; i < end; i+=2) {
        if (primes[i]) { //divergent
            count++;
            for (u32 j = i*i; j <= n; j += i) {
                primes[j] = 0;
            }
        }

        else {
            //all thread NOT PRIME will sleep
        }
    }
}

void better_erathostenes(u32* primes, u32 start, u32 end, u32 n) {
    int tid = threadIDx.x; //each thread handles one element 
    u32 count = 0; //where is this? who owns this?
    __shared__ localstorage[256];
    int local[64];
   
}
    

int main(){
    const int n - 1'000'000;
    int* primes = new int[n];
    for (int i = 0; i < n; i++){
        primes[i] = 1;
    }

    erathostenes<<<(n+31)/32, 32>>>(primes, n);

    return 1;
}