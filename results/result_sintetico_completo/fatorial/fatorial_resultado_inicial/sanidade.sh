#!/bin/bash
# Uso: bash ver_resultados.sh F M V   (sem argumentos: todas as condições)
conds=${@:-V F M H}
for cond in $conds; do
  for d in ${cond}_*/; do
    echo "== ${d%/}"
    f="$d/resultados_catch24_drift/baseline/resultado_otimizacao.json"
    printf "  %-38s %s\n" baseline "$(grep -o '"best_macro_f1": [0-9.]*' "$f" | cut -d' ' -f2)"
    f="$d/resultados_catch24_drift/catch24/resultado_otimizacao.json"
    printf "  %-38s %s\n" catch24 "$(grep -o '"best_macro_f1": [0-9.]*' "$f" | cut -d' ' -f2)"
    for cen in drift catch24_drift; do
      for f in "$d"/resultados_catch24_drift/$cen/*/resultado_otimizacao.json; do
        det=$(basename "$(dirname "$f")")
        printf "  %-38s %s\n" "$cen/$det" "$(grep -o '"best_macro_f1": [0-9.]*' "$f" | cut -d' ' -f2)"
      done
    done
  done
done
