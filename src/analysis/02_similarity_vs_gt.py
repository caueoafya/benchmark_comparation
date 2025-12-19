# -*- coding: utf-8 -*-
"""Similaridade semantica: modelos vs Ground Truth"""

import json
import numpy as np
import pandas as pd
import glob
import os
import re
import unicodedata
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
    if 'query_busca_sugerida' in analysis:
        texts.append(analysis['query_busca_sugerida'])
    return " ".join(texts)


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
gts_invalidos = ['imagem retirada online', 'prompt', 'teste']

resultados = []
print("Processando...")

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
            if not teste_num: continue

            gt_diag, gt_desc = "", ""
            for val in valores:
                val_lower = val.lower()
                if re.match(r'^teste\s*\d+$', val_lower) or any(inv in val_lower for inv in gts_invalidos):
                    continue
                if len(val) < 80 and '?' not in val and not gt_diag:
                    gt_diag = val
                elif ('descri' in val_lower[:15] or len(val) > 80) and '?' not in val[:50]:
                    gt_desc = val

            if not gt_diag and not gt_desc: continue
            gt_completo = f"{gt_diag}. {gt_desc}".strip()
            if len(gt_completo) < 10: continue

            key = f"{pasta_norm}_{teste_num}"
            baseline_r = baseline_idx.get(key)
            if not baseline_r: continue

            baseline_text = extract_response_text(baseline_r)
            gpt4o_text = extract_response_text(gpt4o_idx.get(key))
            gemini_text = extract_response_text(gemini_idx.get(key))

            if not all([baseline_text, gpt4o_text, gemini_text]): continue

            embeddings = model.encode([gt_completo, baseline_text, gpt4o_text, gemini_text])
            gt_emb = embeddings[0].reshape(1, -1)

            resultados.append({
                'pasta': pasta, 'teste': teste_num,
                'sim_baseline': float(cosine_similarity(gt_emb, embeddings[1].reshape(1, -1))[0][0]),
                'sim_gpt4o': float(cosine_similarity(gt_emb, embeddings[2].reshape(1, -1))[0][0]),
                'sim_gemini': float(cosine_similarity(gt_emb, embeddings[3].reshape(1, -1))[0][0])
            })
    except:
        pass

print(f"Total: {len(resultados)} testes")

sim_b = [r['sim_baseline'] * 100 for r in resultados]
sim_g4 = [r['sim_gpt4o'] * 100 for r in resultados]
sim_gm = [r['sim_gemini'] * 100 for r in resultados]

print(f"\nBaseline: {np.mean(sim_b):.1f}%")
print(f"GPT-4o:   {np.mean(sim_g4):.1f}%")
print(f"Gemini:   {np.mean(sim_gm):.1f}%")

output = {
    'total_testes': len(resultados),
    'estatisticas': {
        'baseline': {'media': np.mean(sim_b)},
        'gpt4o_mini': {'media': np.mean(sim_g4)},
        'gemini': {'media': np.mean(sim_gm)}
    },
    'resultados': resultados
}

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\similaridade_vs_gt.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Salvo em: data/processed/similaridade_vs_gt.json")
