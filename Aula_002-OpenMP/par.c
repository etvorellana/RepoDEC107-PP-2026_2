/*
 * DEC107 - Processamento Paralelo (UESC)
 * Exemplo didático: cláusula schedule da diretiva #pragma omp for
 *
 * par.c — versão PARALELA com OpenMP (4 threads)
 *
 * Para cada política de escalonamento, as linhas da matriz 4096x4096 são
 * distribuídas entre as 4 threads. Cada thread pinta com uma cor fixa as
 * linhas que recebeu:
 *     thread 0 -> 32   thread 1 -> 96   thread 2 -> 160   thread 3 -> 228
 *
 * Antes de cada execução a matriz é resetada com 32 (cor da thread master),
 * portanto qualquer linha não atribuída permanece com a cor do fundo.
 *
 * Compilar:  gcc -O2 -fopenmp par.c -o par
 * Executar:  ./par
 * Saídas:    static.pgm, static_chunk16.pgm, dynamic.pgm, dynamic_chunk16.pgm,
 *            guided.pgm, guided_chunk16.pgm, runtime.pgm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define H 4096
#define W 4096
#define NTHREADS 4
#define CHUNK 16   /* tamanho do bloco (chunk) usado nas variantes com chunk */

/* Cores fixas de cada thread (índice = id da thread) */
static const unsigned char CORES[NTHREADS] = {32, 96, 160, 228};

typedef enum {
    SCHED_STATIC,        /* schedule(static)         */
    SCHED_STATIC_CHUNK,  /* schedule(static, CHUNK)  */
    SCHED_DYNAMIC,       /* schedule(dynamic)        */
    SCHED_DYNAMIC_CHUNK, /* schedule(dynamic, CHUNK) */
    SCHED_GUIDED,        /* schedule(guided)         */
    SCHED_GUIDED_CHUNK,  /* schedule(guided, CHUNK)  */
    SCHED_RUNTIME        /* schedule(runtime)        */
} tipo_schedule;

static void salvar_pgm(const char *nome, const unsigned char *mat, int h, int w)
{
    FILE *f = fopen(nome, "wb");
    if (!f) { perror(nome); exit(1); }
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(mat, 1, (size_t)h * w, f);
    fclose(f);
}

/*
 * Corpo do loop, comum a todos os schedules: a thread tid pinta a linha i
 * com a sua cor e atualiza as estatísticas (contagem e intervalo de linhas).
 * cont/min_linha/max_linha são arrays compartilhados, mas cada thread só
 * escreve na sua própria posição [tid] — sem corrida de dados.
 */
#define PINTAR_LINHA(i) do {                                              \
    memset(&mat[(size_t)(i) * W], CORES[tid], W);                         \
    cont[tid]++;                                                          \
    if ((i) < min_linha[tid]) min_linha[tid] = (i);                       \
    if ((i) > max_linha[tid]) max_linha[tid] = (i);                       \
} while (0)

static void executar(unsigned char *mat, const char *nome, tipo_schedule tipo)
{
    /* 1) Reset: fundo com a cor da thread master */
    memset(mat, 32, (size_t)H * W);

    /* 2) Estatísticas por thread */
    int cont[NTHREADS] = {0};
    int min_linha[NTHREADS], max_linha[NTHREADS];
    for (int t = 0; t < NTHREADS; t++) { min_linha[t] = H; max_linha[t] = -1; }

    /* 3) Região paralela: o schedule decide COMO as iterações do for
          (aqui, uma iteração = uma linha) são distribuídas entre as threads */
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(NTHREADS)
    {
        int tid = omp_get_thread_num();

        switch (tipo) {
        case SCHED_STATIC:
            /* Blocos contíguos de ~H/NTHREADS linhas, um por thread,
               atribuídos em round-robin. Determinístico e sem custo. */
            #pragma omp for schedule(static)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;

        case SCHED_STATIC_CHUNK:
            /* Blocos de CHUNK linhas distribuídos ciclicamente:
               thread 0 pega linhas 0-15, 64-79, ... (faixas finas alternadas) */
            #pragma omp for schedule(static, CHUNK)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;

        case SCHED_DYNAMIC:
            /* Blocos de 1 linha (chunk default) atribuídos sob demanda:
               padrão visualmente fragmentado, linha a linha */
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;

        case SCHED_DYNAMIC_CHUNK:
            /* Blocos de CHUNK linhas atribuídos sob demanda: cada thread
               pega o próximo bloco disponível ao terminar o anterior */
            #pragma omp for schedule(dynamic, CHUNK)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;

        case SCHED_GUIDED:
            /* Blocos grandes no início (proporcionais ao trabalho restante),
               diminuindo até 1 linha: faixas largas no topo que afinam */
            #pragma omp for schedule(guided)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;

        case SCHED_GUIDED_CHUNK:
            /* Blocos decrescentes com mínimo de CHUNK linhas */
            #pragma omp for schedule(guided, CHUNK)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;

        case SCHED_RUNTIME:
            /* Política definida pela variável de ambiente OMP_SCHEDULE
               (default: static na maioria das implementações) */
            #pragma omp for schedule(runtime)
            for (int i = 0; i < H; i++) { PINTAR_LINHA(i); }
            break;
        }
    }

    double t1 = omp_get_wtime();

    /* 4) Salva a imagem e imprime as estatísticas */
    salvar_pgm(nome, mat, H, W);

    printf("%-22s tempo: %.4f s\n", nome, t1 - t0);
    for (int t = 0; t < NTHREADS; t++)
        printf("    thread %d: %4d linhas (intervalo %4d..%4d)\n",
               t, cont[t], min_linha[t], max_linha[t]);

    if (tipo == SCHED_RUNTIME) {
        omp_sched_t kind;
        int chunk;
        omp_get_schedule(&kind, &chunk);
        const char *s =
            kind == omp_sched_static  ? "static"  :
            kind == omp_sched_dynamic ? "dynamic" :
            kind == omp_sched_guided  ? "guided"  : "auto";
        printf("    schedule(runtime) efetivo: %s, chunk=%d\n"
               "    (mude com: OMP_SCHEDULE=\"dynamic,4\" ./par)\n", s, chunk);
    }
    printf("\n");
}

int main(void)
{
    unsigned char *mat = malloc((size_t)H * W);
    if (!mat) { perror("malloc"); return 1; }

    omp_set_num_threads(NTHREADS);

    executar(mat, "static.pgm",          SCHED_STATIC);
    executar(mat, "static_chunk16.pgm",  SCHED_STATIC_CHUNK);
    executar(mat, "dynamic.pgm",         SCHED_DYNAMIC);
    executar(mat, "dynamic_chunk16.pgm", SCHED_DYNAMIC_CHUNK);
    executar(mat, "guided.pgm",          SCHED_GUIDED);
    executar(mat, "guided_chunk16.pgm",  SCHED_GUIDED_CHUNK);
    executar(mat, "runtime.pgm",         SCHED_RUNTIME);

    free(mat);
    return 0;
}