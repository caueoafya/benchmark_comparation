# -*- coding: utf-8 -*-
"""Avaliacao multi-dimensional combinando metricas em score ponderado"""

import json
import pandas as pd
import glob
import os
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Carregando modelo de embeddings...")
model = SentenceTransformer('all-mpnet-base-v2')

PESOS = {
    'similaridade_geral': 0.20,
    'diagnostico_direto': 0.25,
    'diagnostico_semantico': 0.25,
    'tipo_imagem': 0.15,
    'achados': 0.15
}

TIPOS_EXAME = ['oct', 'tomografia', 'raio-x', 'radiografia', 'ultrassom', 'ecografia',
               'ressonancia', 'rm', 'tc', 'endoscopia', 'biomicroscopia', 'retinografia',
               'campo visual', 'angiografia', 'fundoscopia', 'dermatoscopia']


def normalize_text(text):
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def extract_response_text(result):
    if not result or not result.get('analysis'):
        return "", "", [], ""

    analysis = result['analysis']
    if 'image_analysis' not in analysis:
        return "", "", [], ""

    img = analysis['image_analysis']
    tipo = img.get('tipo_imagem', '')
    desc = img.get('descricao', '')
    achados = img.get('achados_principais', [])
    if not isinstance(achados, list):
        achados = []

    texto = f"{tipo} {desc} {' '.join(achados)}"
    return tipo, desc, achados, texto


def calc_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    emb = model.encode([text1, text2])
    return max(0, min(1, cosine_similarity([emb[0]], [emb[1]])[0][0]))


def calc_best_match(query, text):
    if not query or not text:
        return 0.0
    frases = [t.strip() for t in re.split(r'[.;,]', text) if len(t.strip()) > 5]
    if not frases:
        return 0.0
    query_emb = model.encode([query])
    texts_emb = model.encode(frases)
    return float(np.max(cosine_similarity(query_emb, texts_emb)[0]))


def evaluate_response(gt_diag, gt_desc, tipo_resp, desc_resp, achados_resp, texto_resp):
    scores = {}

    gt_completo = f"{gt_diag}. {gt_desc}" if gt_desc else gt_diag
    scores['similaridade_geral'] = calc_similarity(gt_completo, texto_resp)

    gt_norm = normalize_text(gt_diag)
    resp_norm = normalize_text(texto_resp)
    if gt_norm in resp_norm:
        scores['diagnostico_direto'] = 1.0
    else:
        palavras = [p for p in gt_norm.split() if len(p) > 3]
        scores['diagnostico_direto'] = sum(1 for p in palavras if p in resp_norm) / len(palavras) if palavras else 0.0

    scores['diagnostico_semantico'] = calc_best_match(gt_diag, texto_resp)

    if gt_desc:
        gt_desc_norm = normalize_text(gt_desc)
        tipo_norm = normalize_text(tipo_resp)
        gt_tipos = [t for t in TIPOS_EXAME if t in gt_desc_norm]
        resp_tipos = [t for t in TIPOS_EXAME if t in tipo_norm]
        if gt_tipos and resp_tipos:
            scores['tipo_imagem'] = len(set(gt_tipos) & set(resp_tipos)) / len(gt_tipos)
        elif not gt_tipos:
            scores['tipo_imagem'] = calc_similarity(gt_desc[:100], tipo_resp) if tipo_resp else 0.5
        else:
            scores['tipo_imagem'] = 0.0
    else:
        scores['tipo_imagem'] = 0.5

    if gt_desc and achados_resp:
        scores['achados'] = calc_similarity(gt_desc, " ".join(achados_resp))
    elif gt_desc:
        scores['achados'] = calc_similarity(gt_desc, desc_resp)
    else:
        scores['achados'] = scores['similaridade_geral']

    scores['score_final'] = sum(scores[k] * PESOS[k] for k in PESOS)
    return scores


# Carregar dados
print("Carregando dados...")
with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\raw\benchmark_results.json", 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)

baseline_results = benchmark_data['results']['gpt-4.1-baseline']
gpt4o_results = benchmark_data['results']['gpt-4o-mini']
gemini_results = benchmark_data['results']['gemini-2.5-flash-lite']


def create_index(results):
    idx = {}
    for r in results:
        parts = r['image_path'].split('/')
        if len(parts) >= 2:
            pasta_norm = normalize_text(parts[0])
            match = re.search(r'(?:teste|caso)\s*[-]?\s*(\d+)', parts[1].lower())
            if match:
                idx[f"{pasta_norm}_{match.group(1)}"] = r
    return idx


baseline_idx = create_index(baseline_results)
gpt4o_idx = create_index(gpt4o_results)
gemini_idx = create_index(gemini_results)

# Carregar GTs
pasta_gt = r"C:\Users\Usuário\Desktop\benchmark_tests\data\ground_truth\GTs de imagem"
planilhas = glob.glob(os.path.join(pasta_gt, "**/*.xlsx"), recursive=True)
gts_invalidos = ['imagem retirada online', 'prompt', 'teste', 'nan']

resultados = []
print("Processando testes...")

for planilha in planilhas:
    pasta = os.path.basename(os.path.dirname(planilha))
    pasta_norm = normalize_text(pasta)

    try:
        df = pd.read_excel(planilha, sheet_name=0)
        for _, row in df.iterrows():
            valores = [str(v).strip() for v in row if pd.notna(v)]

            teste_num = None
            for val in valores:
                match = re.search(r'teste\s*(\d+)', val.lower())
                if match:
                    teste_num = int(match.group(1))
                    break

            if not teste_num:
                continue

            gt_diag, gt_desc = "", ""
            for val in valores:
                val_lower = val.lower().strip()
                if re.match(r'^teste\s*\d+$', val_lower) or any(inv in val_lower for inv in gts_invalidos):
                    continue
                if len(val) < 80 and '?' not in val and not gt_diag and len(val) >= 3:
                    gt_diag = val
                elif (len(val) > 80 or 'descri' in val_lower[:15]) and '?' not in val[:50] and not gt_desc:
                    gt_desc = val

            if not gt_diag:
                continue

            key = f"{pasta_norm}_{teste_num}"
            baseline_r = baseline_idx.get(key)
            if not baseline_r:
                continue

            b = extract_response_text(baseline_r)
            g4 = extract_response_text(gpt4o_idx.get(key))
            gm = extract_response_text(gemini_idx.get(key))

            resultados.append({
                'pasta': pasta, 'teste': teste_num, 'gt_diagnostico': gt_diag,
                'baseline': evaluate_response(gt_diag, gt_desc, *b),
                'gpt4o': evaluate_response(gt_diag, gt_desc, *g4),
                'gemini': evaluate_response(gt_diag, gt_desc, *gm)
            })
    except:
        pass

print(f"Total avaliados: {len(resultados)}")

# Resultados
metricas = list(PESOS.keys()) + ['score_final']
print("\n" + "="*60)
for m in metricas:
    b = np.mean([r['baseline'][m] * 100 for r in resultados])
    g4 = np.mean([r['gpt4o'][m] * 100 for r in resultados])
    gm = np.mean([r['gemini'][m] * 100 for r in resultados])
    print(f"{m:<25} Base:{b:>6.1f}%  GPT4o:{g4:>6.1f}%  Gemini:{gm:>6.1f}%")

# Tempos
tempos = {
    'baseline': np.mean([r['response_time'] for r in baseline_results if r.get('response_time')]),
    'gpt4o_mini': np.mean([r['response_time'] for r in gpt4o_results if r.get('response_time')]),
    'gemini': np.mean([r['response_time'] for r in gemini_results if r.get('response_time')])
}

# Salvar
output = {
    'total_testes': len(resultados),
    'pesos': PESOS,
    'acuracia_final': {
        'baseline': np.mean([r['baseline']['score_final'] * 100 for r in resultados]),
        'gpt4o_mini': np.mean([r['gpt4o']['score_final'] * 100 for r in resultados]),
        'gemini': np.mean([r['gemini']['score_final'] * 100 for r in resultados])
    },
    'tempos': tempos,
    'metricas_detalhadas': {
        m: {
            'baseline': np.mean([r['baseline'][m] * 100 for r in resultados]),
            'gpt4o_mini': np.mean([r['gpt4o'][m] * 100 for r in resultados]),
            'gemini': np.mean([r['gemini'][m] * 100 for r in resultados])
        } for m in metricas
    },
    'resultados_por_teste': resultados
}

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\avaliacao_multi_dimensional.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=float)

print(f"\nSalvo em: data/processed/avaliacao_multi_dimensional.json")
