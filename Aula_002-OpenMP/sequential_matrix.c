#include <stdio.h>
#include <stdlib.h>

#define SIZE 4096

int main() {
    unsigned char *matrix = (unsigned char *)malloc(SIZE * SIZE * sizeof(unsigned char));
    if (matrix == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    // Preenche a matriz com o valor 32
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i * SIZE + j] = 32;
        }
    }

    // Salva a matriz em um arquivo binário
    FILE *fp = fopen("sequential_matrix.bin", "wb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file for writing\n");
        free(matrix);
        return 1;
    }
    fwrite(matrix, sizeof(unsigned char), SIZE * SIZE, fp);
    fclose(fp);

    free(matrix);
    return 0;
}