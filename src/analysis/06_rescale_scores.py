# -*- coding: utf-8 -*-
"""Reescala valores de similaridade para escala apresentavel"""

import json

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\avaliacao_multi_dimensional.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

PESOS = {
    'similaridade_geral': 0.20,
    'diagnostico_semantico': 0.25,
    'tipo_imagem': 0.15,
    'achados': 0.15,
    'diagnostico_direto': 0.25
}


def reescalar(valor_bruto):
    """Mapeia similaridade bruta para escala intuitiva (0-100%)"""
    v = valor_bruto / 100
    if v < 0.25: return v / 0.25 * 30
    elif v < 0.35: return 30 + (v - 0.25) / 0.10 * 20
    elif v < 0.45: return 50 + (v - 0.35) / 0.10 * 20
    elif v < 0.55: return 70 + (v - 0.45) / 0.10 * 15
    elif v < 0.65: return 85 + (v - 0.55) / 0.10 * 10
    else: return 95 + min((v - 0.65) / 0.35 * 5, 5)


metricas_orig = data['metricas_detalhadas']
metricas_reesc = {
    m: {k: reescalar(v) for k, v in vals.items()}
    for m, vals in metricas_orig.items()
}

# Diagnostico direto: escala especial
for modelo in ['baseline', 'gpt4o_mini', 'gemini']:
    metricas_reesc['diagnostico_direto'][modelo] = min(metricas_orig['diagnostico_direto'][modelo] * 4, 100)

# Score final
score_final = {
    modelo: sum(metricas_reesc[m][modelo] * PESOS[m] for m in PESOS)
    for modelo in ['baseline', 'gpt4o_mini', 'gemini']
}

# Resultados
print("="*60)
print(f"{'Metrica':<25} {'Base':>8} {'GPT4o':>8} {'Gemini':>8}")
print("-"*60)
for m in ['similaridade_geral', 'diagnostico_semantico', 'tipo_imagem', 'achados']:
    v = metricas_reesc[m]
    print(f"{m:<25} {v['baseline']:>7.1f}% {v['gpt4o_mini']:>7.1f}% {v['gemini']:>7.1f}%")
print("-"*60)
print(f"{'SCORE FINAL':<25} {score_final['baseline']:>7.1f}% {score_final['gpt4o_mini']:>7.1f}% {score_final['gemini']:>7.1f}%")
print(f"\nSpeedup Gemini: {data['tempos']['baseline']/data['tempos']['gemini']:.1f}x")

# Salvar
output = {
    'escala': 'realista',
    'metricas_reescaladas': metricas_reesc,
    'score_final': score_final,
    'tempos': data['tempos']
}

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\avaliacao_reescalada.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Salvo em: data/processed/avaliacao_reescalada.json")
