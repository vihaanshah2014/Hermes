// training algo practice -- soon to be carried over to proper use cases

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double** allocateMatrix(int rows, int cols){
    double **matrix = malloc(rows * sizeof(double*));

    for(int i = 0; i < rows; i++){
        matrix[i] = malloc(cols * sizeof(double));
    }

    return matrix;
}