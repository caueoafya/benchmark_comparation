# -*- coding: utf-8 -*-
"""Match direto de diagnostico: verifica se GT aparece na resposta"""

import json
import pandas as pd
import glob
import os
import re
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Carregando modelo...")
model = SentenceTransformer('all-mpnet-base-v2')

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\raw\benchmark_results.json", 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)


def normalize_text(text):
    if not text: return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def extract_response_text(result):
    if not result or not result.get('analysis'): return ""
    analysis = result['analysis']
    texts = []
    if 'image_analysis' in analysis:
        img = analysis['image_analysis']
        for key in ['tipo_imagem', 'descricao']:
            if key in img: texts.append(img[key])
        if 'achados_principais' in img and isinstance(img['achados_principais'], list):
            texts.extend(img['achados_principais'])
    return " ".join(texts)


def check_diagnosis(gt_diag, response):
    """Verifica se diagnostico aparece na resposta (direto, palavras-chave ou semantico)"""
    if not gt_diag or not response:
        return False, 0.0, "sem_dados"

    gt_norm = normalize_text(gt_diag)
    resp_norm = normalize_text(response)

    if gt_norm in resp_norm:
        return True, 1.0, "match_direto"

    palavras = [p for p in gt_norm.split() if len(p) > 3]
    if palavras:
        matches = sum(1 for p in palavras if p in resp_norm)
        if matches >= len(palavras) * 0.7:
            return True, 0.8, "match_palavras"

    frases = [f.strip() for f in re.split(r'[.;,]', response) if len(f.strip()) > 10]
    if frases:
        gt_emb = model.encode([gt_diag])
        frases_emb = model.encode(frases)
        max_sim = float(np.max(cosine_similarity(gt_emb, frases_emb)[0]))
        if max_sim >= 0.70:
            return True, max_sim, "match_semantico"
        return False, max_sim, "sem_match"

    return False, 0.0, "sem_frases"


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


baseline_idx = create_index(benchmark_data['results']['gpt-4.1-baseline'])
gpt4o_idx = create_index(benchmark_data['results']['gpt-4o-mini'])
gemini_idx = create_index(benchmark_data['results']['gemini-2.5-flash-lite'])

pasta_gt = r"C:\Users\Usuário\Desktop\benchmark_tests\data\ground_truth\GTs de imagem"
planilhas = glob.glob(os.path.join(pasta_gt, "**/*.xlsx"), recursive=True)
gts_invalidos = ['imagem retirada online', 'prompt', 'teste', 'nan']

resultados = []
print("Processando...")

for planilha in planilhas:
    pasta_norm = normalize_text(os.path.basename(os.path.dirname(planilha)))
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
            if not teste_num: continue

            gt_diag = ""
            for val in valores:
                val_lower = val.lower().strip()
                if re.match(r'^teste\s*\d+$', val_lower) or any(inv in val_lower for inv in gts_invalidos):
                    continue
                if len(val) < 80 and '?' not in val and len(val) >= 3:
                    gt_diag = val
                    break
            if not gt_diag: continue

            key = f"{pasta_norm}_{teste_num}"
            baseline_r = baseline_idx.get(key)
            if not baseline_r: continue

            b_ok, b_conf, b_tipo = check_diagnosis(gt_diag, extract_response_text(baseline_r))
            g4_ok, g4_conf, g4_tipo = check_diagnosis(gt_diag, extract_response_text(gpt4o_idx.get(key)))
            gm_ok, gm_conf, gm_tipo = check_diagnosis(gt_diag, extract_response_text(gemini_idx.get(key)))

            resultados.append({
                'gt': gt_diag,
                'baseline': b_ok, 'gpt4o': g4_ok, 'gemini': gm_ok
            })
    except:
        pass

total = len(resultados)
print(f"Total: {total} testes")

b_acc = sum(1 for r in resultados if r['baseline']) / total * 100
g4_acc = sum(1 for r in resultados if r['gpt4o']) / total * 100
gm_acc = sum(1 for r in resultados if r['gemini']) / total * 100

print(f"\nBaseline: {b_acc:.1f}%")
print(f"GPT-4o:   {g4_acc:.1f}%")
print(f"Gemini:   {gm_acc:.1f}%")

output = {
    'total_testes': total,
    'acuracia': {'baseline': b_acc, 'gpt4o_mini': g4_acc, 'gemini': gm_acc},
    'resultados': resultados
}

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\acuracia_diagnostico_direto.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Salvo em: data/processed/acuracia_diagnostico_direto.json")
