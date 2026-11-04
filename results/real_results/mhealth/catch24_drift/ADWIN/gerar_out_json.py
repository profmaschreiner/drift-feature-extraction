#!/usr/bin/env python3
"""
Gera um arquivo "_out.json" a partir de um resultado_otimizacao.json.

Transformação aplicada:
  - Remove a chave "window_size" de dentro de "best_params".
  - Adiciona essa mesma informação como chave de nível superior
    "fixed_catch24_window_size" (podendo ser sobrescrita via --window-size,
    caso o cenário catch24 utilize um valor de janela fixo diferente do
    que resultou da otimização).

Uso:
    python gerar_out_json.py resultado_otimizacao.json
    python gerar_out_json.py resultado_otimizacao.json --window-size 130
    python gerar_out_json.py resultado_otimizacao.json -o caminho/saida.json
"""

import argparse
import json
from pathlib import Path


def transformar(dados: dict, window_size_fixo: int | None = None) -> dict:
    """Aplica a transformação de best_params.window_size -> fixed_catch24_window_size."""
    dados = dict(dados)  # cópia rasa para não alterar o dict original
    best_params = dict(dados.get("best_params", {}))

    valor_extraido = best_params.pop("window_size", None)
    valor_final = window_size_fixo if window_size_fixo is not None else valor_extraido

    dados["best_params"] = best_params
    if valor_final is not None:
        dados["fixed_catch24_window_size"] = valor_final

    return dados


def gerar_caminho_saida(caminho_entrada: Path) -> Path:
    """Gera o caminho de saída inserindo o sufixo _out antes da extensão."""
    return caminho_entrada.with_name(f"{caminho_entrada.stem}_out{caminho_entrada.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Cria um arquivo _out.json a partir de um resultado_otimizacao.json"
    )
    parser.add_argument("entrada", type=Path, help="Caminho do JSON de entrada")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Caminho do JSON de saída (padrão: <entrada>_out.json)"
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Valor fixo para fixed_catch24_window_size (padrão: usa o window_size da entrada)"
    )
    args = parser.parse_args()

    with open(args.entrada, "r", encoding="utf-8") as f:
        dados = json.load(f)

    dados_transformados = transformar(dados, window_size_fixo=args.window_size)

    caminho_saida = args.output or gerar_caminho_saida(args.entrada)
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(dados_transformados, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"Arquivo gerado: {caminho_saida}")


if __name__ == "__main__":
    main()
