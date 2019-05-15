#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mpi.h"

const int ROW_NUM = 500;
const int COUNT_ITER = 10000;
const int TAG = 0;

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


typedef double DataType;

int N_PROC, PROC_ID;
int BLOCK_FROM, BLOCK_TO;

DataType *data[2];

int *shift;
int *length;
int *blockPrefixLength;

void getBordersTriangle(int curRow, int *from, int *to) {
    *from = ROW_NUM/2 - (int)floor(curRow/2);
    *to = ROW_NUM/2 + (int)floor(curRow/2);
}

DataType getInitValueEdge(int x, int y) {
    return 5;
}

int idx(int curRow, int curShift) {
    int id = curShift - shift[curRow];
    curRow -= BLOCK_FROM;
    id += blockPrefixLength[curRow];
    return id;
}

void calcCurrentBlock() {
    int curBlockSize = (ROW_NUM - 2) / N_PROC;
    if (PROC_ID < (ROW_NUM - 2) % N_PROC) curBlockSize++;

    BLOCK_FROM = (ROW_NUM - 2) / N_PROC * PROC_ID + min((ROW_NUM - 2) % N_PROC, PROC_ID);
    BLOCK_TO = BLOCK_FROM + curBlockSize + 1;
}

int isBorder(int curRow, int curShift) {
    if (curShift == shift[curRow] ||
            curShift == shift[curRow] + length[curRow] - 1)
    {
        return 1;
    }

    if (curRow == 0 || curRow == ROW_NUM - 1) {
        return 1;
    }

    return (curShift < shift[curRow - 1] ||
            curShift >= shift[curRow - 1] + length[curRow - 1] ||
            curShift < shift[curRow + 1] ||
            curShift >= shift[curRow + 1] + length[curRow + 1]);
}

DataType getValue(int index, int curRow, int curShift) {
    return data[index][idx(curRow, curShift)];
}

 void setValue(int index, int curRow, int curShift, DataType val) {
    data[index][idx(curRow, curShift)] = val;
}

void writeMatrixText(FILE *fs, int from, int to) {
    for (int i = from; i < to + 1; i++) {
        for (int j = shift[i]; j < shift[i] + length[i]; ++j) {
            fprintf(fs, "%d %d %.5f\n", i, j, getValue(0, i, j));
        }
    }
}

void writeMatrixBinary(FILE *f, int from, int to) {

    fwrite(
        data[0] + idx(from, shift[from]), sizeof(DataType),
        idx(to + 1, shift[to + 1]) - idx(from, shift[from]), f);
}

void writeMatrix(FILE *f, int from, int to) {
    writeMatrixBinary(f, from, to);
    // writeMatrixText(f, from, to);
}

void writeToFile(const char* output) {
    FILE *fs;

    if (PROC_ID == 0) {
        fs = fopen(output, "w");
        fclose(fs);
    }

    for (int i = 0; i < N_PROC; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (PROC_ID == i) {
            fs = fopen(output, "a");

            if (PROC_ID == 0) {
                writeMatrix(fs, 0, 0);
            }

            writeMatrix(fs, BLOCK_FROM + 1, BLOCK_TO - 1);

            if (PROC_ID == N_PROC - 1) {
                writeMatrix(fs, BLOCK_TO, BLOCK_TO);
            }

            fclose(fs);
        }
    }
}



void solve() {
    DataType *tmp = data[0];

    for (int i = BLOCK_FROM + 1; i < BLOCK_TO; ++i) {

        int l = shift[i];
        int r = l + length[i];

        l = max(l, shift[i - 1]);
        r = min(r, shift[i - 1] + (int)length[i - 1]);

        l = max(l, shift[i + 1]);
        r = min(r, shift[i + 1] + (int)length[i + 1]);

        if (isBorder(i, l)) l++;
        if (isBorder(i, r)) r--;

        for (int j = l; j < r + 1; ++j) {
            DataType newVal = (
                    getValue(0, i - 1, j) + getValue(0, i + 1, j) +
                    getValue(0, i, j - 1) + getValue(0, i, j + 1))
                    / 4.0;
            setValue(1, i, j, newVal);
        }
    }
    
    data[0] = data[1];
    data[1] = tmp;
}

int main(int argc, char **argv) {
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROC_ID);


    shift = (int*)malloc(ROW_NUM * sizeof(int));
    length = (int*)malloc(ROW_NUM * sizeof(int));

    for (int i = 0; i < ROW_NUM; i++) {
        int from, to;
        getBordersTriangle(i, &from, &to);

        shift[i] = from;
        length[i] = to - from + 1;
    }

    calcCurrentBlock();

    blockPrefixLength = (int*)malloc((BLOCK_TO - BLOCK_FROM + 2) * sizeof(int));

    int tt = 0;
    blockPrefixLength[0] = 0;
    for (int i = BLOCK_FROM; i < BLOCK_TO + 1; i++) {
        int id = i - BLOCK_FROM;
        blockPrefixLength[id + 1] = blockPrefixLength[id] + length[i];
        tt += length[i];
    }

    for (int i = 0; i < 2; i++) {
        data[i] = (DataType*) malloc(blockPrefixLength[BLOCK_TO - BLOCK_FROM + 1] * sizeof(DataType));
    }

    for(int i = BLOCK_FROM; i < BLOCK_TO + 1; i++) {
        for(int j = shift[i]; j < shift[i] + length[i]; j++) {
            if (isBorder(i, j)) {
                data[0][idx(i, j)] = getInitValueEdge(i, j);
                data[1][idx(i, j)] = data[0][idx(i, j)];
            }
        }
    }


    writeToFile("data_input.txt");

    int sizeBlock1;
    int sizeBlock2;
    int bufSize;
    MPI_Pack_size(length[BLOCK_FROM + 1], MPI_DOUBLE, MPI_COMM_WORLD, &sizeBlock1);
    MPI_Pack_size(length[BLOCK_TO - 1], MPI_DOUBLE, MPI_COMM_WORLD, &sizeBlock2);
    bufSize = sizeBlock1 + sizeBlock2 + MPI_BSEND_OVERHEAD;
    void *buf = malloc(bufSize);

    if (buf == NULL) {
        printf("%d cannot allocate buf\n", PROC_ID);
        return 1;
    }

    MPI_Buffer_attach(buf, bufSize);
    
    MPI_Status status;

    for(int i = 0; i < COUNT_ITER + 1; i++) {
        solve();

        MPI_Buffer_detach(&buf, &bufSize);
        MPI_Buffer_attach(buf, bufSize);

        if (PROC_ID > 0) {
            MPI_Bsend(data[0] + idx(BLOCK_FROM + 1, shift[BLOCK_FROM+1]), length[BLOCK_FROM + 1], MPI_DOUBLE, PROC_ID - 1, TAG, MPI_COMM_WORLD);
        }
        if (PROC_ID < N_PROC - 1) {
            MPI_Bsend(data[0] + idx(BLOCK_TO - 1, shift[BLOCK_TO - 1]), length[BLOCK_TO - 1], MPI_DOUBLE, PROC_ID + 1, TAG, MPI_COMM_WORLD);
        }


        if (PROC_ID > 0) {
            MPI_Recv(data[0] + idx(BLOCK_FROM, shift[BLOCK_FROM]), length[BLOCK_FROM], MPI_DOUBLE, PROC_ID - 1, TAG, MPI_COMM_WORLD, &status);
        }
        if (PROC_ID < N_PROC - 1) {
            MPI_Recv(data[0] + idx(BLOCK_TO, shift[BLOCK_TO]), length[BLOCK_TO], MPI_DOUBLE, PROC_ID + 1, TAG, MPI_COMM_WORLD, &status);
        }
    }

    char outputFileName[1000];
    sprintf(outputFileName, "data_output_np_%d.txt", N_PROC);
    writeToFile(outputFileName);

    MPI_Buffer_detach(&buf, &bufSize);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
