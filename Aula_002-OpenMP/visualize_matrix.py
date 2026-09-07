#!/usr/bin/env python
"""
DEC107 - Processamento Paralelo (UESC)
visualize_matrix.py — carrega uma matriz binária crua 4096x4096 (uint8)
gerada pelos programas C e exibe como imagem.

Uso:
    python visualize_matrix.py matriz.bin
    python visualize_matrix.py matriz.bin --salvar matriz.png
    python visualize_matrix.py matriz.bin --cmap viridis

Dependências: numpy e matplotlib  (pip install numpy matplotlib)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

H = 4096
W = 4096

# Valores discretos pintados por cada thread no código C (CORES[]).
# Se as cores forem alteradas no C, ajuste esta lista e as fronteiras abaixo.
VALORES_THREADS = [32, 96, 160, 228]
ROTULOS_THREADS = [f"Thread {i} - {v}" for i, v in enumerate(VALORES_THREADS)]
# Fronteiras entre os blocos: cada valor cai no meio do seu intervalo.
FRONTEIRAS = [0, 64, 128, 192, 256]

def ler_matriz(caminho):
    """Lê um arquivo binário cru com H*W bytes uint8 e devolve um array
    NumPy uint8 de forma (H, W)."""
    with open(caminho, 'rb') as f:
        dados = f.read()

    n = H * W
    if len(dados) != n:
        raise ValueError(
            f"{caminho} deve conter exatamente {n} bytes "
            f"(encontrados {len(dados)})")
    return np.frombuffer(dados, dtype=np.uint8).reshape(H, W)

def visualizar_matriz(matriz, titulo, cmap='gray', saida=None):
    # Colormap discreto: amostra o cmap escolhido apenas nos 4 valores de thread
    cmap_base = plt.get_cmap(cmap)
    cmap_discreto = mcolors.ListedColormap(
        cmap_base(np.array(VALORES_THREADS) / 255.0))
    norm_discreta = mcolors.BoundaryNorm(FRONTEIRAS, cmap_discreto.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(matriz, cmap=cmap_discreto, norm=norm_discreta,
               interpolation='nearest')
    barra = plt.colorbar(ticks=VALORES_THREADS, label='valor (cor da thread)')
    barra.ax.set_yticklabels(ROTULOS_THREADS)
    plt.title(titulo)
    plt.tight_layout()

    if saida:
        plt.savefig(saida, dpi=150)
        print(f"Imagem salva em {saida}")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    caminho = sys.argv[1]
    cmap = 'gray'
    saida = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--cmap' and i + 1 < len(args):
            cmap = args[i + 1]
            i += 2
        elif args[i] == '--salvar' and i + 1 < len(args):
            saida = args[i + 1]
            i += 2
        else:
            i += 1

    matriz = ler_matriz(caminho)
    print(f"{caminho}: {matriz.shape[0]} x {matriz.shape[1]}, "
          f"valores presentes: {np.unique(matriz).tolist()}")

    visualizar_matriz(matriz, caminho, cmap=cmap, saida=saida)

if __name__ == '__main__':
    main()