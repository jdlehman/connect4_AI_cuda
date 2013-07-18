/* game.cu
 * Jonathan Lehman
 * April 17, 2012
 *  
 * Compile with: nvcc -o game game.cu
 *
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

__global__ void queen(long*, int);
__device__ void sumBlocks(long *);
void checkArgs(int, char**, int);
void checkGPUCapabilities(int, int, int, int, int);
double getTime();

//set board size
#ifndef _N_
#define _N_ 4
#endif

// Keep track of the gpu time.
cudaEvent_t start, stop; 
float elapsedTime;

// Keep track of the CPU time.
double startTime, stopTime;
 
//array for block sums
long *a;

int main(int argc, char *argv[]){		

	long *dev_a;
	
	//check validity of arguments (should be no arguments)
	checkArgs(argc, argv, 1);
	
	int gW, gH, numberBlocks;
	
	//calculate grid width based on factor N, 
	gW = pow(_N_, numBX);
	
	//depends on if N is even or odd
	int sizePerYSeg = (_N_ / 2) + (_N_ % 2);
	
	gH = sizePerYSeg * numBY;
	
	numberBlocks = gW * gH;	
	
	//check that GPU can handle arguments
	checkGPUCapabilities(gW, gH, _N_, _N_, numberBlocks);
  
	/* Initialize the source arrays here. */
  	a = new long[numberBlocks];  
  	
  	/* Allocate global device memory. */
  	cudaMalloc((void **)&dev_a, sizeof(long) * numberBlocks);
  	
  	/* Start the timer. */
  	cudaEventCreate(&start); 
  	cudaEventCreate(&stop); 
  	cudaEventRecord(start, 0); 
  
  	/* Execute the kernel. */
  	dim3 block(_N_, _N_); //threads w x h
  	dim3 grid(gW, gH); //blocks w x h
  	queen<<<grid, block>>>(dev_a, sizePerYSeg);

  	/* Wait for the kernel to complete. Needed for timing. */  
  	cudaThreadSynchronize();
  	
  	/* Stop the timer and print the resulting time. */
	  cudaEventRecord(stop, 0); 
	  cudaEventSynchronize(stop); 
	  cudaEventElapsedTime(&elapsedTime, start, stop);
	  
  	/* Get result from device. */
  	cudaMemcpy(a, dev_a, sizeof(long) * numberBlocks, cudaMemcpyDeviceToHost); 
  	
  	//print any cuda error messages
  	const char* errorString = cudaGetErrorString(cudaGetLastError());
	printf("GPU Error: %s\n", errorString);
	
	if(sumOnGPU){
		printf("Number of Solutions:%d\n", a[0]);
		  
		//add cpu time and gpu time and print result
		printf( "GPU Time/Total Time: %f secs\n", (elapsedTime / 1000.0));
	}
	else{
	
		/* Start the CPU timer. */
		startTime = getTime();
		
		int sum = 0;
	
		//check if N is even or odd, then calculate sum, which is number of solutions
		if(_N_ % 2 == 0){
			for(int i = 0; i < numberBlocks; i++){ 
				sum+= a[i];
			}
			sum *= 2;
		}
		else{
			int numBlocksPerSeg = numberBlocks / numBY;
			int rowSizeOfGrid = pow(_N_, numBX);
			
			for(int j = 0; j < numBY; j++){
				int start = j * numBlocksPerSeg;
				for(int i = start; i < start + numBlocksPerSeg - rowSizeOfGrid; i++){ 
					sum+= a[i];
				}
			
			}
			sum *= 2;
			
			//add last block row of sums for each Y block
			for(int j = 0; j < numBY; j++){
				for(int i = j * numBlocksPerSeg + numBlocksPerSeg - rowSizeOfGrid; i < j * numBlocksPerSeg + numBlocksPerSeg; i++){ 
					sum+= a[i];
				}
			}
			
		}
		
		/* Stop the CPU timer */
		stopTime = getTime();
		double totalTime = stopTime - startTime;
		  
		printf("Number of Solutions: %d\n", sum);
		  
		//add cpu time and gpu time and print result
		printf( "GPU Time: %f secs\nCPU Time: %f secs\nTotal Time: %f secs\n", (elapsedTime / 1000.0), totalTime, (elapsedTime / 1000.0) + totalTime );
  	}
  	
  	//destroy cuda event
  	cudaEventDestroy(start); 
  	cudaEventDestroy(stop);
    	
  	/* Free the allocated device memory. */
  	cudaFree(dev_a);
  
  	//free allocated host memory
	free(a);
}

__global__
void queen(long *a, int sizePerYSeg){

	__shared__ long solutions[_N_][_N_];
	__shared__ char tuple[_N_][_N_][_N_];
	
	int totalWrong = 0;
	solutions[threadIdx.x][threadIdx.y] = 0;
	
	int totNumGen = powf(_N_, numGen);
	
	int bYsegment = blockIdx.y / sizePerYSeg;
	int workSize = totNumGen / numBY; 
	int extra = totNumGen - workSize * numBY;//extra work to be done by last segment
	
	//set tuple by block Y value
	tuple[threadIdx.x][threadIdx.y][0] = blockIdx.y % sizePerYSeg;
	
	//set tuple(s) by block X value
	int rem = blockIdx.x;
	for(int i = 1; i <= numBX; i++){
		tuple[threadIdx.x][threadIdx.y][i] = rem % _N_;
		rem = rem / _N_;
	}
	
	int tupCtr = numBX;
	
	//set tuples by thread value
	tuple[threadIdx.x][threadIdx.y][++tupCtr] = threadIdx.x;
	tuple[threadIdx.x][threadIdx.y][++tupCtr] = threadIdx.y;
	
	
	
	//check if thread is valid at this point
	for(int i = tupCtr; i > 0; i--){
		for(int j = i - 1, ctr = 1; j >= 0; j--, ctr++){
			//same row
			totalWrong += tuple[threadIdx.x][threadIdx.y][i] == tuple[threadIdx.x][threadIdx.y][j];
			
			//diag upleft
			totalWrong += (tuple[threadIdx.x][threadIdx.y][i] - ctr) == tuple[threadIdx.x][threadIdx.y][j];
			
			//diag downleft
			totalWrong += (tuple[threadIdx.x][threadIdx.y][i] + ctr) == tuple[threadIdx.x][threadIdx.y][j]; 
		}
	}
	
	if(totalWrong == 0){
	
		//iterate through all numbers to generate possible solutions thread must check
		//does not do if thread is already not valid at this point
		int start = bYsegment * workSize;
		for(int c = start; c < start + workSize + (bYsegment == numBY - 1) * extra; c++){
			
			//generate last values in tuple, convert to base N and store to tuple array
			int rem = c;
			for(int b = 0, k = tupCtr + 1; b < numGen; b++, k++){
				tuple[threadIdx.x][threadIdx.y][k] = rem % _N_;
				rem = rem / _N_;
			}
			
			//checks that the numGen tuple values are indeed unique (saves work overall)
			for(int x = 0; x < numGen && totalWrong == 0; x++){
				for(int y = 0; y < numGen && totalWrong == 0; y++){
					totalWrong += tuple[threadIdx.x][threadIdx.y][tupCtr + 1 + x] == tuple[threadIdx.x][threadIdx.y][tupCtr + 1 + y] && x != y;
				}
			}
			
			//check one solution
			for(int i = _N_ - 1; i > totalWrong * _N_; i--){
				for(int j = i - 1, ctr = 1; j >= 0; j--, ctr++){
					//same row
					totalWrong += tuple[threadIdx.x][threadIdx.y][i] == tuple[threadIdx.x][threadIdx.y][j];
					
					//diag upleft
					totalWrong += (tuple[threadIdx.x][threadIdx.y][i] - ctr) == tuple[threadIdx.x][threadIdx.y][j]; 
					
					//diag downleft
					totalWrong += (tuple[threadIdx.x][threadIdx.y][i] + ctr) == tuple[threadIdx.x][threadIdx.y][j];
				}
			}
			
			//add 1 to solution total if nothing wrong
			solutions[threadIdx.x][threadIdx.y] += !(totalWrong);
			
			//reset total wrong
			totalWrong = 0;
		}
	
	}
		
	//sync the threads so that thread 0 can make the calculations
	__syncthreads();
	
	//have thread 0 sum for all threads in block to get block total
	if(threadIdx.x == 0 && threadIdx.y == 0){
	
		//ensure that the block total value is 0 initially
		long sum = 0;
		
		//iterate through each threads solution and add it to the block total
		for(int i =0; i < _N_; i++){
			for(int j = 0; j < _N_; j++){
				//use local var
				sum += solutions[i][j];
			}
		}
		
		//store to global memory
		a[gridDim.x * blockIdx.y + blockIdx.x] = sum;
		
	}
	
	//sync the threads so that calculations can be made
	__syncthreads();
	
	//have the first thread in the first block sum up the block sums to return to the CPU
	if(sumOnGPU == 1 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0){
		sumBlocks(a);
	}
	
} 

__device__
void sumBlocks(long *a){
	long sum = 0;
	int numberBlocks = gridDim.x * gridDim.y;
	int rowSizeOfGrid = powf(_N_, numBX);

	//check if N is even or odd, then calculate sum, which is number of solutions
	if(_N_ % 2 == 0){
		for(int i = 0; i < numberBlocks; i++){ 
			sum+= a[i];
		}
		sum *= 2;
	}
	else{
		int numBlocksPerSeg = numberBlocks / numBY;
		for(int j = 0; j < numBY; j++){
			int start = j * numBlocksPerSeg;
			for(int i = start; i < start + numBlocksPerSeg - rowSizeOfGrid; i++){ 
				sum+= a[i];
			}
		
		}
		sum *= 2;
		
		//add last block row of sums for each Y block
		for(int j = 0; j < numBY; j++){
			for(int i = j * numBlocksPerSeg + numBlocksPerSeg - rowSizeOfGrid; i < j * numBlocksPerSeg + numBlocksPerSeg; i++){ 
				sum+= a[i];
			}
		}
		
	}
	
	//store sum to first index of a
	a[gridDim.x * blockIdx.y + blockIdx.x] = 0;
	a[gridDim.x * blockIdx.y + blockIdx.x] = sum;
	
	
}

void checkArgs(int argc, char *argv[], int numArgs){
	
	//check number of arguments
	if(argc  > numArgs){
		fprintf(stderr, "\nnqueens: Incorrect number of arguments, %d\nCorrect usage: \"nqueens\"\n", argc - 1);
		exit(1);
	}
	
	
	char* invalChar;
	long arg;
	
	//check each argument
	for(int i = 1; i < numArgs; i++){
		//check for overflow of argument
		if((arg = strtol(argv[i], &invalChar, 10)) >= INT_MAX){
			fprintf(stderr, "\nnqueens: Overflow. Invalid argument %d for nqueens, '%s'.\nThe argument must be a valid, positive, non-zero integer less than %d.\n", i, argv[i], INT_MAX);
			exit(1);
		}
	
		//check that argument is a valid positive integer and check underflow
		if(!(arg > 0) || (*invalChar)){
			fprintf(stderr, "\nnqueens: Invalid argument %d for nqueens, '%s'.  The argument must be a valid, positive, non-zero integer.\n", i, argv[i]);
			exit(1);
		}
		
	}	
}

void checkGPUCapabilities(int gridW, int gridH, int blockW, int blockH, int size){
	//check what GPU is being used
	int devId;  
	cudaGetDevice( &devId );
	
	//get device properties for GPU being used
	cudaDeviceProp gpuProp;
	cudaGetDeviceProperties( &gpuProp, devId );
	
	//check if GPU has enough memory 
	if(gpuProp.totalGlobalMem < (size * sizeof(long))){
		fprintf(stderr, "\nnqueens: Insufficient GPU. GPU does not have enough memory to handle the data size: %ld. It can only handle data sizes up to %ld.\n", (size * sizeof(float)) * 3, gpuProp.totalGlobalMem);
		exit(1);
	}
	
	//check if GPU can handle the number of threads per bloc
	if(gpuProp.maxThreadsPerBlock < (blockW * blockH)){
		fprintf(stderr, "\nnqueens: Insufficient GPU. GPU can only handle %d threads per block, not %d.\n", gpuProp.maxThreadsPerBlock, (blockW * blockH));
		exit(1);
	}
	
	//check that GPU can handle the number of threads in the block width
	if(gpuProp.maxThreadsDim[0] < blockW){
		fprintf(stderr, "\nnqueens: Insufficient GPU. GPU can only handle %d threads as the block width of each block, not %d.\n", gpuProp.maxThreadsDim[0], blockW );
		exit(1);
	}
	
	//check that GPU can handle the number of threads in the block height
	if(gpuProp.maxThreadsDim[1] < blockH){
		fprintf(stderr, "\nnqueens: Insufficient GPU. GPU can only handle %d threads as the block height of each block, not %d.\n", gpuProp.maxThreadsDim[1], blockH );
		exit(1);
	}
	
	//check that GPU can handle the number of blocks in the grid width
	if(gpuProp.maxGridSize[0] < gridW){
		fprintf(stderr, "\nnqueens: Insufficient GPU. GPU can only handle %d blocks as the grid width of each grid, not %d.\n", gpuProp.maxGridSize[0], gridW );
		exit(1);
	}
	
	//check that GPU can handle the number of blocks in the grid height
	if(gpuProp.maxGridSize[1] < gridH){
		fprintf(stderr, "\nnqueens: Insufficient GPU. GPU can only handle %d blocks as the grid height of each grid, not %d.\n", gpuProp.maxGridSize[1], gridH );
		exit(1);
	}
}

double getTime(){
  timeval thetime;
  gettimeofday(&thetime, 0);
  return thetime.tv_sec + thetime.tv_usec / 1000000.0;
}

