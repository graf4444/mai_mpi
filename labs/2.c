#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define True 1
#define False 0

#define IDX(i, j, k) ((((k) - FROM) * DIM1 + (i)) * DIM2 + (j))

const int ROOT = 0;
const int TAG = 0;
const int WRITE_TAG = 1;
const int MAX_ITER = 10000;
const double EPS = 0.1;

const int SIZE_DIM1 = 100;
const int SIZE_DIM2 = 100;
const int SIZE_DIM3 = 100;

double * array;
double * auxiliaryArray;

char foutName[100];

int DIM1, DIM2, DIM3;
int N_PROC, PROC_ID;
int FROM, TO;

int min(int a, int b) {
    return a < b ? a : b;
}

void calcCurrentBlock(void) {
    int curBlockSize = (DIM3 - 2) / N_PROC;
    if (PROC_ID < (DIM3 - 2) % N_PROC) curBlockSize++;

    // Process matrices from (DIM1, DIM2, FROM) to (DIM1, DIM2, TO)
    FROM = (DIM3 - 2) / N_PROC * PROC_ID + min((DIM3 - 2) % N_PROC, PROC_ID);
    TO = FROM + curBlockSize + 1;
}


double* allocDataArray(int dataSize) {
    double *data = (double*)malloc(dataSize * sizeof(double));

    if (data == NULL) {
        printf("%ld can't allocate %ld bytes\n",
               PROC_ID + 1, dataSize * sizeof(double));
        MPI_Finalize();
        exit(1);
    }

    return data;
}


void writeMatrixBinary(FILE *f, double* data, int from, int to) {
    fwrite(
        data + IDX(0, 0, from), sizeof(double),
        IDX(0, 0, to + 1) - IDX(0, 0, from), f);
}


void writeMatrixText(FILE *f, double* data, int from, int to) {
    for (int w = from; w <= to; ++w) {
        for (int i = 0; i < DIM1; ++i) {
            for (int j = 0; j < DIM2; ++j) {
                fprintf(f, "%.2f ", data[IDX(i, j, w)]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "==========================\n");
    }
}


void writeMatrix(FILE *f, double *data, int from, int to) {
    // writeMatrixBinary(f, data, from, to);
    writeMatrixText(f, data, from, to);
}

void writeMatrixToFile() {

    FILE *fs;
    MPI_Status st;

    if (PROC_ID == 0) {
        fs = fopen(foutName, "w");
        fclose(fs);
    }

    if (PROC_ID == 0) {
        fs = fopen(foutName, "ab");
        writeMatrix(fs, array, 0, 0);
        fclose(fs);
    }
    
    if (PROC_ID > 0) {
        MPI_Recv(NULL, 0, MPI_DOUBLE, PROC_ID - 1, WRITE_TAG, MPI_COMM_WORLD, &st);
    }

    fs = fopen(foutName, "ab");
    writeMatrix(fs, array, FROM + 1, TO - 1);
    fclose(fs);

    if (PROC_ID < N_PROC - 1) {
        MPI_Send(NULL, 0, MPI_DOUBLE, PROC_ID + 1, WRITE_TAG, MPI_COMM_WORLD);
    }

    if (PROC_ID == (N_PROC - 1)) {
        fs = fopen(foutName, "ab");
        writeMatrix(fs, array, TO, TO);
        fclose(fs);
    }
}

double solve() {
    double localEps = 0.;
    double *tmp = array;

    for (int w = FROM + 1; w <= TO - 1; ++w) {
        for (int i = 1; i < DIM1 - 1; ++i) {
            for (int j = 1; j < DIM2 - 1; ++j) {
                auxiliaryArray[IDX(i, j, w)] = (
                    array[IDX(i + 1, j, w)] + array[IDX(i - 1, j, w)] +
                    array[IDX(i, j + 1, w)] + array[IDX(i, j - 1, w)] +
                    array[IDX(i, j, w + 1)] + array[IDX(i, j, w - 1)]
                ) / 6.0;
                localEps = fmax(localEps, fabs(auxiliaryArray[IDX(i, j, w)] - array[IDX(i, j, w)]));
            }
        }
    }
    
    array = auxiliaryArray;
    auxiliaryArray = tmp;
    return localEps;
}

void initOutfileName(){
    sprintf(foutName, "output_np_%d_l2.txt", N_PROC);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROC_ID);
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);

    double startTime = MPI_Wtime();

    initOutfileName();

    DIM1 = SIZE_DIM1;
    DIM2 = SIZE_DIM2;
    DIM3 = SIZE_DIM3;

    calcCurrentBlock();

    int dataSize = DIM1 * DIM2 * (TO - FROM + 1);

    array = allocDataArray(dataSize);
    auxiliaryArray = allocDataArray(dataSize);

    memset(array, 0, dataSize * sizeof(double));
    memset(auxiliaryArray, 0, dataSize * sizeof(double));

    for (int k = FROM; k <= TO; ++k) {
        for (int i = 0; i < DIM1; ++i) {
            for (int j = 0; j < DIM2; ++j) {
                if (k == 0 || i == 0 || j == 0 ||
                    k == DIM3 - 1 || i == DIM1 - 1 || j == DIM2 - 1)
                {
                    array[IDX(i, j, k)] = (i + j + k);
                    auxiliaryArray[IDX(i, j, k)] = array[IDX(i, j, k)];
                }
            }
        }
    }

    int *sendSizes, *shiftForSend, *recvSizes, *shiftForRecv;
    
    sendSizes = (int*)malloc(N_PROC * sizeof(int));
    shiftForSend = (int*)malloc(N_PROC * sizeof(int));

    recvSizes = (int*)malloc(N_PROC * sizeof(int));
    shiftForRecv = (int*)malloc(N_PROC * sizeof(int));

    memset(sendSizes, 0, N_PROC * sizeof(int));
    memset(shiftForSend, 0, N_PROC * sizeof(int));
    memset(recvSizes, 0, N_PROC * sizeof(int));
    memset(shiftForRecv, 0, N_PROC * sizeof(int));
    
    if (PROC_ID > 0) {
        sendSizes[PROC_ID - 1] = DIM1 * DIM2;
        shiftForSend[PROC_ID - 1] = IDX(0, 0, FROM + 1);
        recvSizes[PROC_ID - 1] = DIM1 * DIM2;
        shiftForRecv[PROC_ID - 1] = IDX(0, 0, FROM);
    }
    if (PROC_ID < N_PROC - 1) {
        sendSizes[PROC_ID + 1] = DIM1 * DIM2;
        shiftForSend[PROC_ID + 1] = IDX(0, 0, TO - 1);
        recvSizes[PROC_ID + 1] = DIM1 * DIM2;
        shiftForRecv[PROC_ID + 1] = IDX(0, 0, TO);
    }
    
    // int sizeBlock;
    // int bufSize;
    // MPI_Pack_size(DIM1 * DIM2, MPI_DOUBLE, MPI_COMM_WORLD, &sizeBlock);
    // bufSize = 2*(sizeBlock + MPI_BSEND_OVERHEAD);
    // void *buf = malloc(bufSize);

    // if (buf == NULL) {
    //     printf("%d cannot allocate buf\n", PROC_ID);
    //     return 1;
    // }

    // MPI_Buffer_attach(buf, bufSize);
    
    // MPI_Status status;
    double* allLocalEps = allocDataArray(N_PROC);
    memset(allLocalEps, 0, N_PROC * sizeof(double));

    double localEps;
    double globalEps;
    int i;
    for (i = 0; i < MAX_ITER; i++) {

        localEps = solve();

        // globalEps = localEps;
        
        MPI_Alltoallv(
            array, sendSizes, shiftForSend, MPI_DOUBLE,
            array, recvSizes, shiftForRecv, MPI_DOUBLE,
            MPI_COMM_WORLD);
        // MPI_Buffer_detach(&buf, &bufSize);
        // MPI_Buffer_attach(buf, bufSize);

        // if (PROC_ID > 0) {
        //     MPI_Bsend(array + IDX(0, 0, FROM + 1), DIM1*DIM2, MPI_DOUBLE, PROC_ID - 1, TAG, MPI_COMM_WORLD);
        // }
        // if (PROC_ID < N_PROC - 1) {
        //     MPI_Bsend(array + IDX(0, 0, TO - 1), DIM1*DIM2, MPI_DOUBLE, PROC_ID + 1, TAG, MPI_COMM_WORLD);
        // }


        // if (PROC_ID > 0) {
        //     MPI_Recv(array, DIM1*DIM2, MPI_DOUBLE, PROC_ID - 1, TAG, MPI_COMM_WORLD, &status);
        // }
        // if (PROC_ID < N_PROC - 1) {
        //     MPI_Recv(array + IDX(0, 0, TO), DIM1*DIM2, MPI_DOUBLE, PROC_ID + 1, TAG, MPI_COMM_WORLD, &status);
        // }
        
        MPI_Allgather(
            &localEps, 1, MPI_DOUBLE,
            allLocalEps, 1, MPI_DOUBLE,
            MPI_COMM_WORLD);

        globalEps = localEps;
        for (int k = 0; k < N_PROC; k++) {
            if (allLocalEps[k] > globalEps) {
                globalEps = allLocalEps[k];
            }
        }
        
        // if (PROC_ID == 0) {
        //     printf("EPS: %f, iter: %d\n", globalEps, i);
        // }
        if (globalEps <= EPS) {
            break;
        }
 
    }

    if (PROC_ID == 0) {
        printf("Iter num: %d\n", i);
        printf("EPS: %f\n", globalEps);

    }

    // MPI_Buffer_detach(&buf, &bufSize);

    writeMatrixToFile();

    double endTime = MPI_Wtime();
    double solveTime = endTime - startTime;

    if (PROC_ID == 0) {
        printf("lead time = %lf\n", solveTime);
    }

    // char fileNameTt[1000];
    // sprintf(fileNameTt, "t_d1_%d_d2_%d_d3_%d.txt", DIM1, DIM2, DIM3);
    // printf("\n\n%s\n\n",fileNameTt);
    // if (PROC_ID == 0 && N_PROC == 1) {
    //     FILE* fs = fopen(fileNameTt, "w");
    //     fclose(fs);
    // }
    // if (PROC_ID == 0) {
    //     FILE* fs = fopen(fileNameTt, "ab");
    //     fprintf(fs, "%d\t%.5f\n", N_PROC, solveTime);
    //     fclose(fs);
    // }

    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}
