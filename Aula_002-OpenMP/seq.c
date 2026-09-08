/*
 * DEC107 - Processamento Paralelo (UESC)
 * Exemplo didático: cláusula schedule da diretiva #pragma omp for
 *
 * seq.c — versão SEQUENCIAL
 * Preenche uma matriz 4096x4096 de unsigned char com o valor 32
 * (cor da thread master) e salva em formato PGM (P5).
 *
 * Compilar:  gcc -O2 -fopenmp seq.c -o seq
 * Executar:  ./seq
 * Saída:     sequencial.pgm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>   /* apenas para omp_get_wtime() */

#define H 4096
#define W 4096

/* Salva a matriz em PGM P5 (Portable GrayMap, binário):
 * cabeçalho ASCII + bytes crus — sem nenhuma biblioteca externa. */
static void salvar_pgm(const char *nome, const unsigned char *mat, int h, int w)
{
    FILE *f = fopen(nome, "wb");
    if (!f) { perror(nome); exit(1); }
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(mat, 1, (size_t)h * w, f);
    fclose(f);
}

int main(void)
{
    /*
     * Alocação única e layout ROW-MAJOR: mat[i*W + j] é a célula da linha i,
     * coluna j. É o layout natural do C e mantém cada linha contígua na
     * memória — o que será importante na versão paralela (localidade e
     * menos false sharing entre threads).
     */
    unsigned char *mat = malloc((size_t)H * W);
    if (!mat) { perror("malloc"); return 1; }

    double t0 = omp_get_wtime();
    memset(mat, 32, (size_t)H * W);   /* tudo com a cor da thread master */
    double t1 = omp_get_wtime();

    printf("Sequencial: preencheu %d x %d com o valor 32 em %.4f s\n",
           H, W, t1 - t0);

    salvar_pgm("sequencial.pgm", mat, H, W);
    printf("Arquivo gerado: sequencial.pgm\n");

    free(mat);
    return 0;
}