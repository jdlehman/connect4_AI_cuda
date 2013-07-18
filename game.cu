/* game.cu
 * Jonathan Lehman
 * April 17, 2012
 *  
 * Compile with: nvcc -o game game.cu
 *
 */

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <sys/time.h>

#define DEBUG 0    /* set DEBUG to flag to 1 for debugging, set flag to 0 for optimized performance */
#define ROWS 6   /* number of rows in connect-4 */
#define COLS 7   /* number of columns in connect-4 */
#define RED -1   /* value for red tokens and red player */
#define BLACK 1  /* value for black tokens and black player */
#define HUMANPLAYER 1  /* by default we assume that human player plays black */

#define numB 5 /*power number of blocks in grid raised to (for width and height each) */
#define numTX 2
#define numTY 1

using namespace std;

typedef struct 
{
   int board[ROWS][COLS];  /* -1 token red player, 1 token black player, 0 empty square */
   int currentplayer;  /* -1 red player, 1 black player */
   int tokensonboard;  /* counts the number of tokens on the board */
}Game;

typedef struct
{
   int row;    /* row coordinate of square */
   int col;    /* column coordinate of square */
   int token;  /* -1 red token, 1 black token */
} Move;

/* The game state (struct Game *game) is passed as a reference 
   parameter (pointer) to the functions in order to avoid that the 
   entire board state is copied with each function call */

void InitGame(Game *game);  /* set up empty board and initial player */
void MakeMove(Game *game, Move move);  /* make a move */
void UndoMove(Game *game, Move move);  /* undo a move */
int Win(Game *game, int player);  /* checks if player (red=-1, black=1) has won game */  
int Draw(Game *game);  /* checks if game ended in a draw */  
void PossibleMoves(Game *game, int *number_of_moves, Move moves[]); /* computes the possible moves , number_of_moves returns the number of available moves, moves[] contains list of moves */ 
void DisplayBoard(Game *game); /* print board state on screen */
void DisplayMove(Move move); /* print move on screen */
//int Utility(struct Game *game);  /* returns the Utility of a non-terminal board state */
void EnterMove(Move *move, Game *game); /* reads in a move from the keyboard */
int Row(Game *game, int col); /* computes the row on which token ends when dropped in column col */

__global__ void generate(Game, Move*);
__device__ int Evaluate(Game*);
__device__ int Win2(Game *game, int player);  /* checks if player (red=-1, black=1) has won game */  
__device__ int Draw2(Game *game);  /* checks if game ended in a draw */ 
void checkGPUCapabilities(int, int, int, int, int);

// Keep track of the gpu time.
cudaEvent_t start, stop; 
float elapsedTime;

int main(int argc, char *argv[])
{
  int i;
  Game game;
  Move moves[COLS];
  //int number_of_moves;
  int playagainsthuman=0; /* computer plays against itself (0) or against human (1) */ 
  
  for (i=1; i<argc; i++)  /* iterate through all command line arguments */
  {    
    if(strcmp(argv[i],"-p")==0)  /* if command line argument -p human opponent */
      {
	playagainsthuman=1; 
	printf("Human player plays black\n");
      }
    if(strcmp(argv[i],"-h")==0)  /* if command line argument -h print help */
      {
	printf("game [-p] [-h]\n-p for play against human player\n-h for help\n");
        return 0;  /* quit program */
      }
  }
  
  InitGame(&game);   /* set up board */
  while( !Draw(&game) && !Win(&game,RED) && !Win(&game,BLACK))   /* no draw or win */
   {
      //int rand;
      Move move;
      Move *compMove = NULL;
      DisplayBoard(&game);  /* display board state */
      
      //PossibleMoves(&game,&number_of_moves,moves); /* calculate available moves */
      //rand = (int) (drand48()*number_of_moves);  /* pick a random move */
      //MakeMove(&game,moves[rand]); /* make move */
      
      
      
      
      
      
      
      
      //CUDA STUFF
      Game dev_game = game;
      Move *dev_move = NULL;
      //dev_move.row = 0;
      //dev_move.col = 0;
	
      int threadX = 49;
      int threadY = 7;
      
      int gridSize = pow(7, numB);
	
	//check that GPU can handle arguments
	checkGPUCapabilities(gridSize, gridSize, threadX, threadY, gridSize * gridSize);
  	
  	/* Allocate global device memory. */
  	cudaMalloc((void **)&(dev_game.board), sizeof(int) * COLS * ROWS);
	cudaMalloc((void **)&dev_move, sizeof(Move) );
  	
  	/* Start the timer. */
  	cudaEventCreate(&start); 
  	cudaEventCreate(&stop); 
  	cudaEventRecord(start, 0); 
  
  	/* Execute the kernel. */
  	dim3 block(threadX, threadY); //threads w x h
  	dim3 grid(gridSize, gridSize); //blocks w x h

  	generate<<<grid, block>>>(dev_game, dev_move);//passes current game config, and empty shell to store best move

  	
	/* Wait for the kernel to complete. Needed for timing. */  
  	cudaDeviceSynchronize();
	
  	/* Stop the timer and print the resulting time. */
	  cudaEventRecord(stop, 0); 
	  cudaEventSynchronize(stop); 
	  cudaEventElapsedTime(&elapsedTime, start, stop);
	  
  	/* Get move result from device. */
  	//cudaMemcpy(compMove, dev_move, sizeof(Move), cudaMemcpyDeviceToHost); 

  	//print any cuda error messages
  	const char* errorString = cudaGetErrorString(cudaGetLastError());
	printf("hi6\n");
	printf("GPU Error: %s\n", errorString);
	
	//printf("Moveee: %d %d\n", (*compMove).row, (*compMove).col);
  	
  	//destroy cuda event
  	//cudaEventDestroy(start); 
  	//cudaEventDestroy(stop);
    	
  	/* Free the allocated device memory. */
  	//cudaFree(dev_move);
	cudaFree(dev_game.board);
  
  	//free allocated host memory
	//free(a);
      
      
      
      
      
      
      
      
      
      
      
      MakeMove(&game,*compMove);  /* make move */
      DisplayMove(*compMove); /* display move */

      if (playagainsthuman) /* human player */
	{
	  DisplayBoard(&game); /* show board state after computer moved */
          if (!Draw(&game) && !Win(&game,RED)) /* no draw and no computer win */
	    {
	      EnterMove(&move,&game); /* human player enters her move */
	      MakeMove(&game,move);  /* make move */
	    }
	} /* end of if humanplayer */
   }  /* end of while not draw or win */

  DisplayBoard(&game);  /* display board state */
  if (Draw(&game))
   printf("the game ended in a draw\n");
  if (Win(&game, RED))
   printf("player red won the game\n");
  if (Win(&game, BLACK))
   printf("player black won the game\n");
  return 0;
}  /* end of main */

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

void InitGame(Game *game)
{
  int i;
  int j;
  for (i=0; i < ROWS; i++)
    for (j=0; j < COLS; j++)	
      (*game).board[i][j]=0;  /* empty board */
  (*game).currentplayer=RED;   /* red player to start game */  
  (*game).tokensonboard=0; 
};

void MakeMove(Game *game, Move move)
{

#if DEBUG
   assert((*game).board[move.row][move.col]==0); /* assert square is empty */
   if (move.row>0)
      assert((*game).board[move.row-1][move.col]!=0); /* assert square below is occupied */
   assert((*game).currentplayer==move.token);  /* assert that right player moves */
#endif
   (*game).board[move.row][move.col]=move.token;  /* place token at square */
   (*game).currentplayer*=-1; /* switch player */
   (*game).tokensonboard++; /* increment number of tokens on board by one */
}
      
void UndoMove(Game *game, Move move)
{

#if DEBUG
   assert((*game).board[move.row][move.col]!=0); /* assert square is occupied */
   if (move.row<ROWS-1)
      assert((*game).board[move.row+1][move.col]==0); /* assert square above is empty */
   assert((*game).currentplayer!=move.token);  /* assert that right player moves */
#endif
   (*game).board[move.row][move.col]=0;  /* remove token from square */
   (*game).currentplayer*=-1; /* switch player */
   (*game).tokensonboard--; /* decrement number of tokens on board by one */
}

int Draw(Game *game)
{
  if ((*game).tokensonboard<42)
    return 0;
  else return (!Win(game,RED) && !Win(game,BLACK));
}

int Win(Game *game, int player)
{
   int i;
   int j;
   for (j=0;j<COLS;j++)
     for(i=0;i<ROWS-3;i++) /* check for vertical four tokens in a column */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i+count][j]==player))  /* check if token is ok and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a column */
	    return 1;   /* win for player */
        }

   for (j=0;j<COLS-3;j++)
     for(i=0;i<ROWS;i++) /* check for four horizontal tokens in a row */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i][j+count]==player))  /* check if token is ok and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a row */
	    return 1;   /* win for player */
        }

  for (j=0;j<COLS-3;j++)
     for(i=0;i<ROWS-3;i++) /* check for four tokens in an upward diagonal */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i+count][j+count]==player))  /* check if token is owned by player and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a diagonal */
	    return 1;   /* win for player */
        }

  for (j=0;j<COLS-3;j++)
     for(i=3;i<ROWS;i++) /* check for four tokens in a downward diagonal */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i-count][j+count]==player))  /* check if token is owned by player and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a diagonal */
	    return 1;   /* win for player */
        }

  return 0;  /* no win for player */
}

void PossibleMoves(Game *game, int *number_of_moves, Move moves[])
 /* computes the possible moves , 
    number_of_moves returns the number of available moves, 
    moves[] contains list of moves */ 
{
  int i;
  *number_of_moves=0;
  for (i=0;i<COLS;i++)
   {
     int row=Row(game,i); /* computes first empty square in col i */
     if (row<ROWS) /* column has an empty square */
      {
        moves[*number_of_moves].row=row;
        moves[*number_of_moves].col=i;
        moves[*number_of_moves].token=(*game).currentplayer;
        (*number_of_moves)++;
      }
   }
}


void DisplayBoard(Game *game) /* print board state on screen */
{
   int i;
   int j;
   for(i=ROWS-1;i>=0;i--)	
   {
      for (j=0;j<COLS;j++)
       switch ((*game).board[i][j])
	{
           case -1:
           printf("X|");
           break;
	  case 1:
           printf("0|");
           break;
          case 0:
           printf(" |");
           break;
        }
      printf("\n");
   }
   printf("--------------\n0|1|2|3|4|5|6\n\n");
}

 
void DisplayMove(Move move) /* print move on screen */
{
   if (move.token==-1)
    printf("X->(%d,%d)\n",move.row,move.col);
   else
    printf("0->(%d,%d)\n",move.row,move.col);	
}


void EnterMove(Move *move, Game *game) /* reads in a move from the keyboard */
{
  int col=0;
  int row=0;
  do
  {
      do 
      {
        printf("\nEnter the column [0-6] of token: ");
        scanf("%d",&col);
      } while ((col < 0) || (col >= COLS));
      row=Row(game,col);
      if (row >= ROWS)
        printf("column %d is full\n", col);
  } while(row>=ROWS);
  printf("your move 0->(%d,%d)\n",row,col);
  (*move).row=row;
  (*move).col=col;
  (*move).token=HUMANPLAYER;
}

int Row(Game *game, int col) /* computes the row on which token ends when dropped in column col */
{
  int row=0;
  while((row<ROWS) && (*game).board[row][col]!=0)
      row++;
  return row;
}

__device__
int Evaluate(Game *game)
{
  if (Draw2(game))
    return 0;
  if (Win2(game,RED))
    return 10; /* maximum utility for winning */
  if (Win2(game,BLACK))
    return -10; /* minimum utility for losing */

  return 0;
}

__device__
int Draw2(Game *game)
{
  if ((*game).tokensonboard<42)
    return 0;
  else return (!Win2(game,RED) && !Win2(game,BLACK));
}

__device__
int Win2(Game *game, int player)
{
   int i;
   int j;
   for (j=0;j<COLS;j++)
     for(i=0;i<ROWS-3;i++) /* check for vertical four tokens in a column */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i+count][j]==player))  /* check if token is ok and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a column */
	    return 1;   /* win for player */
        }

   for (j=0;j<COLS-3;j++)
     for(i=0;i<ROWS;i++) /* check for four horizontal tokens in a row */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i][j+count]==player))  /* check if token is ok and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a row */
	    return 1;   /* win for player */
        }

  for (j=0;j<COLS-3;j++)
     for(i=0;i<ROWS-3;i++) /* check for four tokens in an upward diagonal */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i+count][j+count]==player))  /* check if token is owned by player and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a diagonal */
	    return 1;   /* win for player */
        }

  for (j=0;j<COLS-3;j++)
     for(i=3;i<ROWS;i++) /* check for four tokens in a downward diagonal */
       {
          int count=0;  /* counts number of consecutive tokens */
	  while((count < 4) && ((*game).board[i-count][j+count]==player))  /* check if token is owned by player and not 4 tokens yet */
	    count++;
          if (count==4) /* four tokens in a diagonal */
	    return 1;   /* win for player */
        }

  return 0;  /* no win for player */
}


__global__
void generate(Game game, Move *move){

	/*__shared__ int evalVals[49][7];//array to store each threads evaluation
	__shared__ char moves[49][7][42];
	
	int mvCtr = 0;
		
	 int start = mvCtr;
	 int end = mvCtr + numB;
	 mvCtr = end;
	int rem = blockIdx.x;
	for(int i = start; i < end; i++){
		moves[threadIdx.x][threadIdx.y][i] = rem % 7;
		rem = rem / 7;
	}
	
	rem = blockIdx.y;
	 start = mvCtr;
	 end = mvCtr + numB;
	 mvCtr = end;
	for(int i = start; i < end; i++){
		moves[threadIdx.x][threadIdx.y][i] = rem % 7;
		rem = rem / 7;
	}
	
	rem = threadIdx.x;
	 start = mvCtr;
	 end = mvCtr + numTX;
	 mvCtr = end;
	for(int i = start; i < end; i++){
		moves[threadIdx.x][threadIdx.y][i] = rem % 7;
		rem = rem / 7;
	}
	
	moves[threadIdx.x][threadIdx.y][mvCtr] = threadIdx.y;*/
      
	
	(*move).col = 2;
	(*move).row = 1;
	(*move).token = 1;
	
	
	/*if(totalWrong == 0){
	
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
	}*/
	
} 
