#!/usr/bin/env python
"""
DEC107 - Processamento Paralelo (UESC)
visualizar.py — carrega um PGM P5 gerado pelos programas C e exibe como imagem.

Uso:
    python visualizar.py static.pgm
    python visualizar.py dynamic_chunk16.pgm --salvar dynamic.png
    python visualizar.py guided.pgm --cmap viridis

Dependências: numpy e matplotlib  (pip install numpy matplotlib)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Valores discretos pintados por cada thread no código C (CORES[]).
# Se as cores forem alteradas no C, ajuste esta lista e as fronteiras abaixo.
VALORES_THREADS = [32, 96, 160, 228]
ROTULOS_THREADS = [f"Thread {i} - {v}" for i, v in enumerate(VALORES_THREADS)]
# Fronteiras entre os blocos: cada valor cai no meio do seu intervalo.
FRONTEIRAS = [0, 64, 128, 192, 256]

def ler_pgm(caminho):
    """Lê um PGM P5 (binário) e devolve um array NumPy uint8 de forma (H, W)."""
    with open(caminho, 'rb') as f:
        dados = f.read()

    # Cabeçalho PGM: 4 tokens ASCII separados por espaços/quebras de linha
    # (comentários #... permitidos): magic, largura, altura, maxval.
    pos = 0
    tokens = []
    while len(tokens) < 4:
        while pos < len(dados) and dados[pos] in b' \t\r\n':
            pos += 1
        if dados[pos:pos + 1] == b'#':              # comentário
            while pos < len(dados) and dados[pos] not in b'\r\n':
                pos += 1
            continue
        inicio = pos
        while pos < len(dados) and dados[pos] not in b' \t\r\n':
            pos += 1
        tokens.append(dados[inicio:pos])

    magic, largura, altura, maxval = tokens
    if magic != b'P5':
        raise ValueError(f"{caminho} não é um PGM P5 (magic={magic!r})")
    w, h = int(largura), int(altura)

    pos += 1  # pula o único caractere de espaço entre o cabeçalho e os dados

    n = w * h
    bloco = dados[pos:pos + n]
    if len(bloco) != n:
        raise ValueError(f"arquivo truncado: cabeçalho promete {n} bytes")
    return np.frombuffer(bloco, dtype=np.uint8).reshape(h, w)

def visualizar_matriz(matriz, titulo, cmap='gray', saida=None):
    # Colormap discreto: amostra o cmap escolhido apenas nos 4 valores de thread
    cmap_base = plt.get_cmap(cmap)
    cmap_discreto = mcolors.ListedColormap(
        cmap_base(np.array(VALORES_THREADS) / 255.0))
    norm_discreta = mcolors.BoundaryNorm(FRONTEIRAS, cmap_discreto.N)

    plt.figure(figsize=(6, 6))
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

    arr = ler_pgm(caminho)
    print(f"{caminho}: {arr.shape[0]} x {arr.shape[1]}, "
          f"valores presentes: {np.unique(arr).tolist()}")

    visualizar_matriz(arr, caminho, cmap=cmap, saida=saida)

if __name__ == '__main__':
    main()