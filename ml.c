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

void freeMatrix(double **matrix, int rows){
    for(int i = 0; i < rows; i++){
        free(matrix[i]);
    }
    free(matrix);
}

//Gauss Jorden Elimincation to compute the inverse of a matrix

void matrixInverse(double **a, int n, double **inverse){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            inverse[i][j] = (i == j) ? 1 : 0;
        }
    }

    for(int i = 0; i < n; i++){
        double pivot = a[i][i];
        for(int j = 0; j < n; j++){
            a[i][j] /= pivot;
            inverse[i][j] /= pivot;
        }
        for(int j = 0; j < n; j++){
            if(i != j){
                double factor = a[j][i];
                for(int k = 0; k < n; k++){
                    a[j][k] -= factor * a[i][k];
                    inverse[j][k] -= factor * inverse[i][k];
                }
            }
        }
    }

}