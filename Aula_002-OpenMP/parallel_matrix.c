#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 4096

int main() {
    unsigned char *matrix = (unsigned char *)malloc(SIZE * SIZE * sizeof(unsigned char));
    if (matrix == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    int thread_ids[] = {32, 96, 160, 228};

    #pragma omp parallel num_threads(4) default(none) shared(matrix, thread_ids)
    {
        int id = omp_get_thread_num();
        int chunk_size = SIZE / 4;
        int start = id * chunk_size;
        int end = (id == 3) ? SIZE : (id + 1) * chunk_size;

        for (int i = start; i < end; i++) {
            for (int j = 0; j < SIZE; j++) {
                matrix[i * SIZE + j] = thread_ids[id];
            }
        }
    }

    // Salva a matriz em um arquivo binário
    FILE *fp = fopen("parallel_matrix.bin", "wb");
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