/* c4.cu
* Jonathan Lehman
* April 18, 2012
*
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>

using namespace std;

//macros user can change
#define width 7 //board width
#define height 6 //board height
#define toWin 4 //number in a row needed to win
#define maxDepth 4 //max depth for cpu recursive alg to search (for CPU calc)
#define useGPU 1 //boolean to determine if run comp move search on GPU (default false)

//macros user shouldn't change
#define numB 3 //number moves generate per block id (if gpu used)
#define numTX 2 //number moves generate per thread id x (if gpu used)
#define numTY 1 //number moves generate per thread id y (if gpu used)
#define tx width * width //number of threads in x dir
#define ty width //number threads in y dir
#define numMoves 2 * numB + numTX + numTY //number of total moves generated per thread

#define maxMoves width * height //used for determining game end by draw
#define winScore 10000
#define invalidScore 10000000

//board object to hold all necessary information about
//current board config
typedef struct{
    int moveScore[width];//array scores of moves
    char square[width][height];//array of element at each board square
    int lastRow, lastCol;//last row and col played (for purpose of undoing)
    int totalPieces;
}Board;

//function prototypes
void init(Board &board);
void checkArgs(int argc, char *argv[], int numArgs);
void printBoard(Board &board);
char getSquare(Board &board, int col, int row);
void doTurn(Board &board, int playerType, char currentPlayer);
void humanTurn(Board &board, char currentPlayer);
void compTurn(Board &board, char currentPlayer);
char checkWin(Board &board);
void changePlayer(char &currentPlayer);
int canMove(Board &board, int col);
void doMove(Board &board, int col, char currentPlayer);
int isDraw(Board &board);
void undoMove(Board &board, int col, char currentPlayer);
int evaluate(Board &board);
void checkGPUCapabilities(int, int, int, int, int);

//computer logic
int determineMove(Board& board, char player);
int alphabeta(Board& board, char player, int alpha, int beta, int depth);

//cuda functions
__global__ void generateMove(Board *board, int *move, char currentPlayer);
__device__ void transferBoard(Board *newBoard, Board *oldBoard);
__device__ void devDoMove(Board &board, int col, char currentPlayer);
__device__ int devCanMove(Board &board, int col);
__device__ void devChangePlayer(char &currentPlayer);
__device__ int devEvaluate(Board &board);
__device__ char devCheckWin(Board &board);
__device__ int devIsDraw(Board &board);
__device__ char devGetSquare(Board &board, int col, int row);
__device__ int devDetermineMove(Board& board, char player);
template<int> __device__ int devAlphabeta(Board& board, char player, int alpha, int beta);
__device__ void devUndoMove(Board &board, int col, char currentPlayer);

//Keep track of the gpu time.
cudaEvent_t start, stop; 
float elapsedTime;
int humanPlayers;

int main(int argc, char *argv[]){
    
    //check arguments
    checkArgs(argc, argv, 2);
    
    //create and initialize board
    Board board;
    init(board);
    
    printf("\nConnect 4 Game:\n\nInitial Board:\n\n");
    printBoard(board);
    
    //represents players, 0 for computer, 1 for human player
    int player1, player2;
    
    if(humanPlayers == 0){
        player1 = 0;
        player2 = 0;
    }
    else if(humanPlayers == 1){
        player1 = 1;//human goes first
        player2 = 0;
    }
    else{
        player1 = 1;
        player2 = 1;
    }
    
    char winner;
    char currentPlayer = 'X';//X goes first
    
    //loop while players move
    while(1){
        //player 1 turn
        doTurn(board, player1, currentPlayer);
        if(isDraw(board) || (winner = checkWin(board)) != ' '){//break on draw or win
            break;
        }
        
        //change players
        changePlayer(currentPlayer);
        
        //player 2 turn
        doTurn(board, player2, currentPlayer);
        if(isDraw(board) || (winner = checkWin(board)) != ' '){//break on draw or win
            break;
        }
        
        //change players
        changePlayer(currentPlayer);
    }
    
    //do something with winner
    if(winner != ' '){
        printf("\nGame Over.\nPlayer %c wins!\n", winner);
    }
    else{
        printf("\nGame Over.\nIt's a draw.\n");
    }
    
}

//check arguments (should allow user to set the number of players)
void checkArgs(int argc, char *argv[], int numArgs){
   
    //check number of arguments
    if(argc != numArgs){
            fprintf(stderr, "\nIncorrect number of arguments, %d\nCorrect usage: \"c4 [0-2]\",\nwhere the number specified between 0 and 2 is the number of human players\n", argc - 1);
            exit(1);
    }
    
    //check first argument
    char* invalChar;
    long arg;

    //convert first argument to int
    arg = strtol(argv[1], &invalChar, 10);

    //check that first argument is between 0 and 2 and an int value
    if((arg < 0) || (arg > 2) || (*invalChar)){
        fprintf(stderr, "\nInvalid argument for c4, '%s'.\nThe argument must be an integer between 0 and 2 inclusive.\n", argv[1]);
        exit(1);
    }
    
    //set number of human players
    humanPlayers = arg;
    	
}

//initialize the game board
void init(Board &board){
    //set last moves to 0
    board.lastCol = board.lastRow = 0;
    //set number of pieces on board to 0
    board.totalPieces = 0;

    //set each board square to blank
    for(int col = 0; col < width; col++){
        
        for(int row = 0; row < height; row++){
            board.square[col][row] = ' ';
        }
        //set move score to 0
        board.moveScore[col] = 0;
    }
}

//print board to terminal
void printBoard(Board &board){

    //print elements on board
    for(int row = height - 1; row >= 0; row--){
        printf("|");
        for(int col = 0; col < width; col++){                
            printf("%c|", getSquare(board, col, row));
        }
        printf("\n");
    }
 
    for(int col = 0; col < width; col++){
        printf("--");
    }
    
    //print bottom of board with column numbers
    printf("-\n|");
    for(int col = 0; col < width; col++){
        printf("%d|", col);
    }
    printf("\n\n");
}

//gets character at specific square on board
char getSquare(Board &board, int col, int row){
    return board.square[col][row];
}

//do turn based on player type (human or computer)
void doTurn(Board &board, int playerType, char currentPlayer){
    //state who's move it is
    if(currentPlayer == 'X'){
        printf("Player 1's Turn (X):\n");
    }
    else{
        printf("Player 2's Turn (O):\n");
    }
    
    //determine whether to prompt for move, or generate computer move
    if(playerType){//human
        humanTurn(board, currentPlayer);
    }
    else{//computer
        compTurn(board, currentPlayer);
    }
}

//do human turn
void humanTurn(Board &board, char currentPlayer){
    int move;
    
    //prompt user for move, accept only if valid
    do{
        string str;
        printf("\nPlease enter a valid column (0-6) as your move:\n");
        getline(cin, str);
        stringstream(str) >> move;
    }while(move < 0 || move >= width || !canMove(board, move));

    //make move
    doMove(board, move, currentPlayer);
    printBoard(board);
}

//do computer turn
void compTurn(Board &board, char currentPlayer){
    int move;
    
    //get computer move
    if(!useGPU){
        move = determineMove(board, currentPlayer);
    }
    else{
        do{
            //make cuda kernel call
            Board *cudaBoard;
            int *cudaMove;
                    
    
            //int threadX = tx;
            //int threadY = ty;
            //int gridSize = pow(width, numB);
            
            //allocate memory on GPU device
            cudaMalloc((void**)&cudaBoard, sizeof(Board));
            cudaMalloc((void**)&cudaMove, sizeof(int));
            
            //copy board to device
            cudaMemcpy(cudaBoard, &board, sizeof(Board), cudaMemcpyHostToDevice);
            
            //check that GPU can handle arguments
            //checkGPUCapabilities(gridSize, gridSize, threadX, threadY, gridSize * gridSize);
            
            /* Start the timer. */
            cudaEventCreate(&start); 
            cudaEventCreate(&stop); 
            cudaEventRecord(start, 0);
      
            /* Execute the kernel. */
            dim3 block(7, 7); //threads w x h
            dim3 grid(1, 1); //blocks w x h
    
            //passes current board config and current player, and empty shell to store best move
            generateMove<<<grid, block>>>(cudaBoard, cudaMove, currentPlayer);
            
            /* Wait for the kernel to complete. Needed for timing. */  
            cudaThreadSynchronize();//or device sync? apparently thread sync is outdated
            
            /* Stop the timer and print the resulting time. */
            cudaEventRecord(stop, 0); 
            cudaEventSynchronize(stop); 
            cudaEventElapsedTime(&elapsedTime, start, stop);
    
            //retrieve the results
            cudaMemcpy(&move, cudaMove, sizeof(int), cudaMemcpyDeviceToHost);
            
            //print any cuda error messages
            const char* errorString = cudaGetErrorString(cudaGetLastError());
            printf("GPU Error: %s\n", errorString);
            
            //print gpu time
            printf("GPU Time: %f secs\n", (elapsedTime / 1000.0));
            
            //printf("move = %d\n", move);
            //destroy cuda event
            cudaEventDestroy(start); 
            cudaEventDestroy(stop);
            
            /* Free the allocated device memory. */
            cudaFree(cudaMove);
            cudaFree(cudaBoard);
        }while(move < 0 || move > width);//in case something weird happens on GPU
    }
    
    //do move and print results
    doMove(board, move, currentPlayer);
    printf("\nComputer put piece in column %d\n", move);
    printBoard(board);
}

//check if most recent move has caused a win
//return winners character piece, or blank for no win
char checkWin(Board &board){
    //only check near most recently placed piece
    //to see if it causes a win
    
    int col1,row1,col2,row2;
    char player = getSquare(board, board.lastCol, board.lastRow);
    
    //check for horizontal win
    col1 = col2 = board.lastCol;
    //check right
    while(col1 < width && getSquare(board, col1, board.lastRow) == player){
        col1++;
    }
    //Go left
    while(col2 >= 0 && getSquare(board, col2, board.lastRow) == player){
        col2--;
    }
    //check 4 in a row
    if(col1 - col2 > toWin){
        return player;
    }

    //check for a vertical win
    row1 = row2 = board.lastRow;
    //check up
    while(row1 < height && getSquare(board, board.lastCol, row1) == player){
        row1++;
    }
    //check down
    while(row2 >= 0 && getSquare(board, board.lastCol, row2) == player){
        row2--;
    }
    //check 4 in a row
    if(row1 - row2 > toWin){
        return player;
    }

    //check southeast/northwest diagonal win
    col1 = col2 = board.lastCol;
    row1 = row2 = board.lastRow;
    //check southeast
    while(row1 >= 0 && col1 < width && getSquare(board, col1, row1) == player){
        col1++;
        row1--;
    }
    //check northwest
    while(row2 < height && col2 >= 0 && getSquare(board, col2, row2) == player) {
        col2--;
        row2++;
    }
    //check 4 in a row
    if(col1 - col2 > toWin){
        return player;
    }

    //check for northeast/southwest win
    col1 = col2 = board.lastCol;
    row1 = row2 = board.lastRow;
    //check southwest
    while(row1 >= 0 && col1 >= 0 && getSquare(board, col1, row1) == player){
        col1--;
        row1--;
    }
    //check northeast
    while(row2 < height && col2 < width && getSquare(board, col2, row2) == player){
        col2++;
        row2++;
    }
    //check 4 in a row
    if(col2 - col1 > toWin){
        return player;
    }

    //no winner, return blank
    return ' ';
}

//change players
void changePlayer(char &currentPlayer){
    if(currentPlayer == 'X'){
        currentPlayer = 'O';
    }
    else{
        currentPlayer = 'X';
    }
}

//check if a move can be made in colum col
int canMove(Board &board,  int col){
    return board.square[col][height - 1] == ' ';
}

//make move on board (at colum col with piece current player)
void doMove(Board &board, int col, char currentPlayer){
    //iterate through row in column and place in first empty spot
    for(int row = 0; row < height; row++){
        if(getSquare(board, col, row) == ' '){
            //set data
            board.square[col][row] = currentPlayer;
            board.lastCol = col;
            board.lastRow = row;
            board.totalPieces++;//increment number of pieces
            return;
        }
    }
}

//checks if game is over, assuming win has not been made at this point
//so draw
int isDraw(Board &board){
    return board.totalPieces >= maxMoves;
}

//undo last move
void undoMove(Board &board,  int col, char currentPlayer){
    //remove last piece placed in row
    int row = height-1;
    //iterate down row in column piece was placed in until find piece
    while (row >= 0 && getSquare(board, col, row) == ' '){
        row--;
    }
    if (getSquare(board, col, row) == currentPlayer){
        board.square[col][row] = ' ';
    }
    
    //decrement total
    board.totalPieces--;
}


//returns best move out of possible moves (uses alphabeta function as subroutine)
int determineMove(Board &board, char currentPlayer){

    //player X turn, maximize
    if(currentPlayer == 'X'){
        
        //iterate through moves and get scores
        int maxScore = -invalidScore;
        int maxMove  = 0;
        for (int move = 0; move < width; move++)
            if(canMove(board, move)){
                doMove(board, move, 'X');
                int score = alphabeta(board, 'O', -invalidScore, invalidScore, 0);

                board.moveScore[move] = score;

                if(score >= maxScore){
                    maxScore = score;
                    maxMove = move;
                }
                undoMove(board,move,'X');
            }
            else{
                //set move score to invalid score if can't move there
                board.moveScore[move] = invalidScore;
            }

        //return move with highest score
        return maxMove;
    }

    //player O turn, minimize
    else if(currentPlayer == 'O'){
        
        //iterate through moves and get scores
        int minScore = invalidScore;
        int minMove  = 0;
        for(int move = 0; move < width; move++){
            if(canMove(board, move)){
                doMove(board, move, 'O');
                int score = alphabeta(board, 'X', -invalidScore, invalidScore, 0);
                
                board.moveScore[move] = score;

                if(score < minScore){
                    minScore = score;
                    minMove = move;
                }
                undoMove(board,move,'O');
            }
            else{
                //set move score to invalid score if can't move there
                board.moveScore[move] = invalidScore;
            }
        }
        //Return the move with the least score
        return minMove;
    }
    else{
        //never gets called
        return 0;
    }
}

//returns highest score based on possible moves
int alphabeta(Board& board, char player, int alpha, int beta, int depth){
    //check if win
    char winner = checkWin(board);
    if(winner == 'X'){
        return winScore;
    }
    else if(winner == 'O'){
        return -winScore;
    }

    if(depth >= maxDepth || isDraw(board)){
        //return score of winless board
        return evaluate(board);
    }

    //player X turn, will maximize
    if(player == 'X'){
        //iterate through moves and get scores
        //int maxScore = -invalidScore;
        for(int move = 0; move < width; move++)
            if(canMove(board, move))
            {
                doMove(board, move, 'X');
                int score = alphabeta(board, 'O', alpha, beta, depth + 1);
                undoMove(board, move, 'X');
                if(score > alpha){
                    alpha = score;
                }
                if(alpha >= beta){
                    return alpha;
                }
            }
        return alpha;
    }

    //player O turn, minimize
    else if(player == 'O'){
        
        //iterate through moves and get scores
        //int minScore = invalidScore;
        for(int move = 0; move < width; move++)
            if(canMove(board, move)){
                doMove(board, move, 'O');
                int score = alphabeta(board, 'X', alpha, beta, depth + 1);
                undoMove(board,move,'O');
                if(score < beta){
                    beta = score;
                }
                if(alpha >= beta){
                    return beta;                
                }
            }
        return beta;
    }
    else{
        //never gets called
        return 0;
    }
}

//evaluation function used to get a score when the game does not reach the end
int evaluate(Board &board){
    int score = 0;

    //Score for each position
    //middle is favorable, because more 4 in a row possibilites
    //and blocks more of opponents
    //|1|2|3|4|3|2|1|
    //|2|3|4|5|4|3|2|
    //|3|4|5|6|5|4|3|
    //|2|3|4|5|4|3|2|
    //|1|2|3|4|3|2|1|
    //|0|1|2|3|2|1|0|

    //iterate through columns
    for(int col = 0; col < width; col++){
        int colScore = (width / 2) - col;
        
        //make positive if negative
        if (colScore < 0){
            colScore = -colScore;
        }
        colScore = (width / 2) - colScore;

        //Count the number of pieces in each column
        //and score accordingly
        for(int row = 0; row < height; row++){
            int rowScore = (height / 2) - row;
            
            //make positive if negative
            if(rowScore < 0){
                rowScore = -rowScore;
            }
            rowScore = (height / 2) - rowScore;

            if(getSquare(board, col, row) == 'X'){
                score += colScore + rowScore;
            }
            else if(getSquare(board, col, row) == 'O'){
                score -= colScore + rowScore;                
            }
        }
    }
    
    return score;
}

void checkGPUCapabilities(int gridW, int gridH, int blockW, int blockH, int size){
	//check what GPU is being used
	int devId;  
	cudaGetDevice( &devId );
	
	//get device properties for GPU being used
	cudaDeviceProp gpuProp;
	cudaGetDeviceProperties( &gpuProp, devId );
	
	//check if GPU has enough memory 
	if(gpuProp.totalGlobalMem < (size * sizeof(int))){
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

//cuda kernel to generate best move for computer
__global__
void generateMove(Board *board, int *move, char currentPlayer){
    __shared__ int score[width][width];
    __shared__ int moveScore[width];
    
    Board newBoard;
    transferBoard(&newBoard, board);
    //*move = devDetermineMove(newBoard, currentPlayer);
    
    //check if can do move
    if(devCanMove(newBoard, threadIdx.x)){
        devDoMove(newBoard, threadIdx.x, currentPlayer);
        //change player
        devChangePlayer(currentPlayer);
        //score[threadIdx.x] = devDetermineMove(newBoard, currentPlayer);
    }
    else{
        score[threadIdx.x][threadIdx.y] = -invalidScore;
    }
    
    if(devCanMove(newBoard, threadIdx.y)){
        devDoMove(newBoard, threadIdx.y, currentPlayer);
        //change player
        devChangePlayer(currentPlayer);
        if(currentPlayer == 'X'){
            score[threadIdx.x][threadIdx.y] = devDetermineMove(newBoard, currentPlayer);
        }
        else{
            score[threadIdx.x][threadIdx.y] = -devDetermineMove(newBoard, currentPlayer);
        }
        
    }
    else{
        score[threadIdx.x][threadIdx.y] = -invalidScore;
    }
    
    //compare in thread 0
    if(threadIdx.x == 0 && threadIdx.y == 0){
        for(int i = 0; i < width; i++){
            moveScore[i] = 0;
            for(int j = 0; j < width; j++){
                if(score[i][j] == i){
                    moveScore[i]++;
                }
            }
        }
        
        int maxScore = -invalidScore;
        int theMove;
        for(int i = 0; i < width; i++){
            if(moveScore[i] > maxScore){
                maxScore = moveScore[i];
                theMove = i;
            }
        }
        *move = theMove;
    }
    
}


/*cuda device functions*/

//copy data from old board to new board
__device__
void transferBoard(Board *newBoard, Board *oldBoard){
    
    (*newBoard).lastRow = (*oldBoard).lastRow;
    (*newBoard).lastCol = (*oldBoard).lastCol;
    (*newBoard).totalPieces = (*oldBoard).totalPieces;
    
    for(int col = 0; col < width; col++){
        (*newBoard).moveScore[col] = (*oldBoard).moveScore[col];
        
        for(int row = 0; row < height; row++){
            (*newBoard).square[col][row] = (*oldBoard).square[col][row];
        }
    }
}

//check if a move can be made in colum col
__device__
int devCanMove(Board &board,  int col){
    return board.square[col][height - 1] == ' ';
}

//make move on board (at colum col with piece current player)
__device__
void devDoMove(Board &board, int col, char currentPlayer){
    //iterate through row in column and place in first empty spot
    for(int row = 0; row < height; row++){
        if(devGetSquare(board, col, row) == ' '){
            //set data
            board.square[col][row] = currentPlayer;
            board.lastCol = col;
            board.lastRow = row;
            board.totalPieces++;//increment number of pieces
            return;
        }
    }
}

//change players
__device__
void devChangePlayer(char &currentPlayer){
    if(currentPlayer == 'X'){
        currentPlayer = 'O';
    }
    else{
        currentPlayer = 'X';
    }
}

//evaluation function used to get a score when the game does not reach the end
__device__
int devEvaluate(Board &board){
    int score = 0;

    //Score for each position
    //middle is favorable, because more 4 in a row possibilites
    //and blocks more of opponents
    //|1|2|3|4|3|2|1|
    //|2|3|4|5|4|3|2|
    //|3|4|5|6|5|4|3|
    //|2|3|4|5|4|3|2|
    //|1|2|3|4|3|2|1|
    //|0|1|2|3|2|1|0|

    //iterate through columns
    for(int col = 0; col < width; col++){
        int colScore = (width / 2) - col;
        
        //make positive if negative
        if (colScore < 0){
            colScore = -colScore;
        }
        colScore = (width / 2) - colScore;

        //Count the number of pieces in each column
        //and score accordingly
        for(int row = 0; row < height; row++){
            int rowScore = (height / 2) - row;
            
            //make positive if negative
            if(rowScore < 0){
                rowScore = -rowScore;
            }
            rowScore = (height / 2) - rowScore;

            if(devGetSquare(board, col, row) == 'X'){
                score += colScore + rowScore;
            }
            else if(devGetSquare(board, col, row) == 'O'){
                score -= colScore + rowScore;                
            }
        }
    }
    
    return score;
}

//gets character at specific square on board
__device__
char devGetSquare(Board &board, int col, int row){
    return board.square[col][row];
}

//check if most recent move has caused a win
//return winners character piece, or blank for no win
__device__
char devCheckWin(Board &board){
    //only check near most recently placed piece
    //to see if it causes a win
    
    int col1,row1,col2,row2;
    char player = devGetSquare(board, board.lastCol, board.lastRow);
    
    //check for horizontal win
    col1 = col2 = board.lastCol;
    //check right
    while(col1 < width && devGetSquare(board, col1, board.lastRow) == player){
        col1++;
    }
    //Go left
    while(col2 >= 0 && devGetSquare(board, col2, board.lastRow) == player){
        col2--;
    }
    //check 4 in a row
    if(col1 - col2 > toWin){
        return player;
    }

    //check for a vertical win
    row1 = row2 = board.lastRow;
    //check up
    while(row1 < height && devGetSquare(board, board.lastCol, row1) == player){
        row1++;
    }
    //check down
    while(row2 >= 0 && devGetSquare(board, board.lastCol, row2) == player){
        row2--;
    }
    //check 4 in a row
    if(row1 - row2 > toWin){
        return player;
    }

    //check southeast/northwest diagonal win
    col1 = col2 = board.lastCol;
    row1 = row2 = board.lastRow;
    //check southeast
    while(row1 >= 0 && col1 < width && devGetSquare(board, col1, row1) == player){
        col1++;
        row1--;
    }
    //check northwest
    while(row2 < height && col2 >= 0 && devGetSquare(board, col2, row2) == player) {
        col2--;
        row2++;
    }
    //check 4 in a row
    if(col1 - col2 > toWin){
        return player;
    }

    //check for northeast/southwest win
    col1 = col2 = board.lastCol;
    row1 = row2 = board.lastRow;
    //check southwest
    while(row1 >= 0 && col1 >= 0 && devGetSquare(board, col1, row1) == player){
        col1--;
        row1--;
    }
    //check northeast
    while(row2 < height && col2 < width && devGetSquare(board, col2, row2) == player){
        col2++;
        row2++;
    }
    //check 4 in a row
    if(col2 - col1 > toWin){
        return player;
    }

    //no winner, return blank
    return ' ';
}

//checks if game is over, assuming win has not been made at this point
//so draw
__device__
int devIsDraw(Board &board){
    return board.totalPieces >= maxMoves;
}


//returns best move out of possible moves (uses alphabeta function as subroutine)
__device__
int devDetermineMove(Board &board, char currentPlayer){

    //player X turn, maximize
    if(currentPlayer == 'X'){
        
        //iterate through moves and get scores
        int maxScore = -invalidScore;
        int maxMove  = 0;
        for (int move = 0; move < width; move++)
            if(devCanMove(board, move)){
                devDoMove(board, move, 'X');
                int score = devAlphabeta<0>(board, 'O', -invalidScore, invalidScore);

                board.moveScore[move] = score;

                if(score >= maxScore){
                    maxScore = score;
                    maxMove = move;
                }
                devUndoMove(board,move,'X');
            }
            else{
                //set move score to invalid score if can't move there
                board.moveScore[move] = invalidScore;
            }

        //return move with highest score
        return maxMove;
    }

    //player O turn, minimize
    else if(currentPlayer == 'O'){
        
        //iterate through moves and get scores
        int minScore = invalidScore;
        int minMove  = 0;
        for(int move = 0; move < width; move++){
            if(devCanMove(board, move)){
                devDoMove(board, move, 'O');
                int score = devAlphabeta<0>(board, 'X', -invalidScore, invalidScore);
                
                board.moveScore[move] = score;

                if(score < minScore){
                    minScore = score;
                    minMove = move;
                }
                devUndoMove(board,move,'O');
            }
            else{
                //set move score to invalid score if can't move there
                board.moveScore[move] = invalidScore;
            }
        }
        //Return the move with the least score
        return minMove;
    }
    else{
        //never gets called
        return 0;
    }
}

template<>
__device__ int devAlphabeta<maxDepth>(Board& board, char player, int alpha, int beta){
    return devEvaluate(board);
}

//returns highest score based on possible moves
template <int depth>
__device__
int devAlphabeta(Board& board, char player, int alpha, int beta){
    //check if win
    char winner = devCheckWin(board);
    if(winner == 'X'){
        return winScore;
    }
    else if(winner == 'O'){
        return -winScore;
    }

    if(devIsDraw(board)){
        //return score of winless board
        return devEvaluate(board);
    }

    //player X turn, will maximize
    if(player == 'X'){
        //iterate through moves and get scores
        //int maxScore = -invalidScore;
        for(int move = 0; move < width; move++)
            if(devCanMove(board, move))
            {
                devDoMove(board, move, 'X');
                int score = devAlphabeta<depth+1>(board, 'O', alpha, beta);
                devUndoMove(board, move, 'X');
                if(score > alpha){
                    alpha = score;
                }
                if(alpha >= beta){
                    return alpha;
                }
            }
        return alpha;
    }

    //player O turn, minimize
    else if(player == 'O'){
        
        //iterate through moves and get scores
        //int minScore = invalidScore;
        for(int move = 0; move < width; move++)
            if(devCanMove(board, move)){
                devDoMove(board, move, 'O');
                int score = devAlphabeta<depth+1>(board, 'X', alpha, beta);
                devUndoMove(board,move,'O');
                if(score < beta){
                    beta = score;
                }
                if(alpha >= beta){
                    return beta;                
                }
            }
        return beta;
    }
    else{
        //never gets called
        return 0;
    }
}

//undo last move
__device__
void devUndoMove(Board &board,  int col, char currentPlayer){
    //remove last piece placed in row
    int row = height-1;
    //iterate down row in column piece was placed in until find piece
    while (row >= 0 && devGetSquare(board, col, row) == ' '){
        row--;
    }
    if (devGetSquare(board, col, row) == currentPlayer){
        board.square[col][row] = ' ';
    }
    
    //decrement total
    board.totalPieces--;
}