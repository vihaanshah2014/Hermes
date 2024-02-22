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

// code out later -------------------------------->


// Multiply two matrices
void matrixMultiply(double **a, int aRows, int aCols, double **b, int bRows, int bCols, double **result) {
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            result[i][j] = 0;
            for (int k = 0; k < aCols; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Transpose a matrix
void matrixTranspose(double **a, int rows, int cols, double **transpose) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transpose[j][i] = a[i][j];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("error\n");
        return 1;
    }
    FILE *trainFile = fopen(argv[1], "r");
    FILE *dataFile = fopen(argv[2], "r");
    
    if (!trainFile || !dataFile) {
        printf("error\n");
        return 1;
    }

    char fileType[6];
    
    int k, n, m;
    fscanf(trainFile, "%5s", fileType);
    if (strcmp(fileType, "train") != 0) {
        printf("error\n");
        return 1;
    }
    
    fscanf(trainFile, "%d", &k);
    fscanf(trainFile, "%d", &n);
    
    double **X = allocateMatrix(n, k + 1);
    double **Y = allocateMatrix(n, 1);
    
    for (int i = 0; i < n; i++) {
        X[i][0] = 1; // First column of X is all ones
        for (int j = 1; j <= k; j++) {
            fscanf(trainFile, "%lf", &X[i][j]);
        }
        fscanf(trainFile, "%lf", &Y[i][0]);
    }


    fclose(trainFile);

    fscanf(dataFile, "%5s", fileType);
    if (strcmp(fileType, "data") != 0) {
        printf("error\n");
        return 1;
    }
    int inputK;
    
    fscanf(dataFile, "%d", &inputK);
    if (inputK != k) {
        printf("error\n");
        return 1;
    }
    fscanf(dataFile, "%d", &m);
    
    double **inputData = allocateMatrix(m, k + 1);
    for (int i = 0; i < m; i++) {
        inputData[i][0] = 1; // First column is all ones
        for (int j = 1; j <= k; j++) {
            fscanf(dataFile, "%lf", &inputData[i][j]);
        }
    }
    
    fclose(dataFile);

    double **X_T = allocateMatrix(k + 1, n);
    matrixTranspose(X, n, k + 1, X_T);

    double **product = allocateMatrix(k + 1, k + 1);
    matrixMultiply(X_T, k + 1, n, X, n, k + 1, product);

    double **inverse = allocateMatrix(k + 1, k + 1);
    matrixInverse(product, k + 1, inverse);

    double **weights = allocateMatrix(k + 1, 1);
    double **intermediate = allocateMatrix(k + 1, n);

    matrixMultiply(inverse, k + 1, k + 1, X_T, k + 1, n, intermediate);

    matrixMultiply(intermediate, k + 1, n, Y, n, 1, weights);

    freeMatrix(intermediate, k + 1);


    double **prices = allocateMatrix(m, 1);
    matrixMultiply(inputData, m, k + 1, weights, k + 1, 1, prices);

    for (int i = 0; i < m; i++) {
        printf("%.0f\n", prices[i][0]);
    }

    freeMatrix(X, n);
    freeMatrix(Y, n);
    freeMatrix(X_T, k + 1);
    freeMatrix(product, k + 1);
    freeMatrix(inverse, k + 1);
    freeMatrix(weights, k + 1);
    freeMatrix(inputData, m);
    freeMatrix(prices, m);
    
    return 0;
}