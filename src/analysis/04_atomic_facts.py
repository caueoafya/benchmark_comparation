# -*- coding: utf-8 -*-
"""Avaliacao por fatos atomicos: cobertura de informacoes do GT"""

import json
import pandas as pd
import glob
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Carregando modelo...")
model = SentenceTransformer('all-mpnet-base-v2')

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\raw\benchmark_results.json", 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)


def create_index(results):
    return {r['image_path']: r for r in results}


baseline_idx = create_index(benchmark_data['results']['gpt-4.1-baseline'])
gpt4o_idx = create_index(benchmark_data['results']['gpt-4o-mini'])
gemini_idx = create_index(benchmark_data['results']['gemini-2.5-flash-lite'])


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


def extract_facts(text):
    if not text: return []
    facts = re.split(r'[.;:]', re.sub(r'\s+', ' ', text.strip()))
    result = []
    for fact in facts:
        fact = fact.strip()
        if ',' in fact and len(fact) > 80:
            result.extend([f.strip() for f in fact.split(',') if len(f.strip()) > 10])
        elif len(fact) > 5:
            result.append(fact)
    return result


def calc_coverage(gt_facts, resp_facts, threshold=0.55):
    if not gt_facts or not resp_facts: return 0.0
    gt_emb = model.encode(gt_facts)
    resp_emb = model.encode(resp_facts)
    sim_matrix = cosine_similarity(gt_emb, resp_emb)
    matched = sum(1 for i in range(len(gt_facts)) if np.max(sim_matrix[i]) >= threshold)
    return matched / len(gt_facts)


def check_diagnosis(gt_diag, response, threshold=0.45):
    if not gt_diag or not response: return False, 0.0
    gt_emb = model.encode([gt_diag.lower()])
    parts = [p.strip() for p in re.split(r'[.,;:\n]', response.lower()) if len(p.strip()) > 3]
    if not parts: return False, 0.0
    resp_emb = model.encode(parts)
    max_sim = float(np.max(cosine_similarity(gt_emb, resp_emb)[0]))
    return max_sim >= threshold, max_sim


pasta_gt = r"C:\Users\Usuário\Desktop\benchmark_tests\data\ground_truth\GTs de imagem"
planilhas = glob.glob(os.path.join(pasta_gt, "**/*.xlsx"), recursive=True)

gts = {}
for planilha in planilhas:
    pasta = os.path.basename(os.path.dirname(planilha))
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

            diag, desc = "", ""
            for val in valores:
                if len(val) < 80 and '?' not in val and not diag:
                    diag = val
                elif len(val) > 80:
                    desc = val

            if diag and desc:
                gts[f"{pasta}/Teste {teste_num}"] = {
                    'diag': diag, 'desc': desc, 'facts': extract_facts(desc)
                }
    except:
        pass

print(f"GTs carregados: {len(gts)}")

resultados = []
for path in baseline_idx.keys():
    parts = path.split('/')
    if len(parts) < 2: continue

    match = re.search(r'(?:teste|caso)\s*(\d+)', parts[1].lower())
    if not match: continue

    chave = f"{parts[0]}/Teste {int(match.group(1))}"
    if chave not in gts: continue

    gt = gts[chave]
    b_resp = extract_response_text(baseline_idx.get(path))
    g4_resp = extract_response_text(gpt4o_idx.get(path))
    gm_resp = extract_response_text(gemini_idx.get(path))

    b_diag, _ = check_diagnosis(gt['diag'], b_resp)
    g4_diag, _ = check_diagnosis(gt['diag'], g4_resp)
    gm_diag, _ = check_diagnosis(gt['diag'], gm_resp)

    resultados.append({
        'path': path,
        'baseline_diag': b_diag,
        'baseline_facts': calc_coverage(gt['facts'], extract_facts(b_resp)),
        'gpt4o_diag': g4_diag,
        'gpt4o_facts': calc_coverage(gt['facts'], extract_facts(g4_resp)),
        'gemini_diag': gm_diag,
        'gemini_facts': calc_coverage(gt['facts'], extract_facts(gm_resp))
    })

print(f"Avaliados: {len(resultados)}")

b_diag_acc = sum(1 for r in resultados if r['baseline_diag']) / len(resultados) * 100
g4_diag_acc = sum(1 for r in resultados if r['gpt4o_diag']) / len(resultados) * 100
gm_diag_acc = sum(1 for r in resultados if r['gemini_diag']) / len(resultados) * 100

b_fact_avg = np.mean([r['baseline_facts'] for r in resultados]) * 100
g4_fact_avg = np.mean([r['gpt4o_facts'] for r in resultados]) * 100
gm_fact_avg = np.mean([r['gemini_facts'] for r in resultados]) * 100

print(f"\nDiagnostico - Base: {b_diag_acc:.1f}% | GPT4o: {g4_diag_acc:.1f}% | Gemini: {gm_diag_acc:.1f}%")
print(f"Cobertura   - Base: {b_fact_avg:.1f}% | GPT4o: {g4_fact_avg:.1f}% | Gemini: {gm_fact_avg:.1f}%")


def to_native(obj):
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_native(i) for i in obj]
    return obj


output = {
    'total': len(resultados),
    'estatisticas': {
        'baseline': {'diag': b_diag_acc, 'facts': b_fact_avg},
        'gpt4o_mini': {'diag': g4_diag_acc, 'facts': g4_fact_avg},
        'gemini': {'diag': gm_diag_acc, 'facts': gm_fact_avg}
    },
    'resultados': to_native(resultados)
}

with open(r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\avaliacao_fatos_atomicos.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Salvo em: data/processed/avaliacao_fatos_atomicos.json")
