# -*- coding: utf-8 -*-
"""Similaridade semantica entre modelos usando Sentence Transformers (all-mpnet-base-v2)"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

print("Carregando modelo de embeddings...")
model = SentenceTransformer('all-mpnet-base-v2')

json_path = r"C:\Users\Usuário\Desktop\benchmark_tests\data\raw\benchmark_results.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total de testes: {data['metadata']['total_tests']}")

baseline_results = data['results']['gpt-4.1-baseline']
gpt4o_mini_results = data['results']['gpt-4o-mini']
gemini_results = data['results']['gemini-2.5-flash-lite']


def extract_text_from_analysis(analysis_dict):
    """Extrai texto relevante de uma analise para comparacao"""
    if not analysis_dict or not isinstance(analysis_dict, dict):
        return ""

    texts = []
    if 'analysis' in analysis_dict and analysis_dict['analysis']:
        analysis = analysis_dict['analysis']
        if 'image_analysis' in analysis:
            img = analysis['image_analysis']
            if 'tipo_imagem' in img:
                texts.append(f"Tipo de imagem: {img['tipo_imagem']}")
            if 'descricao' in img:
                texts.append(f"Descricao: {img['descricao']}")
            if 'achados_principais' in img:
                achados = img['achados_principais']
                if isinstance(achados, list):
                    texts.append(f"Achados: {'; '.join(achados)}")
        if 'query_busca_sugerida' in analysis:
            texts.append(f"Query: {analysis['query_busca_sugerida']}")
    return " ".join(texts)


def calculate_semantic_similarity(text1, text2):
    """Calcula similaridade semantica via cosine similarity"""
    if not text1 or not text2:
        return 0.0
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return max(0, min(1, similarity))


def create_index_by_path(results):
    return {r['image_path']: r for r in results}


baseline_index = create_index_by_path(baseline_results)
gpt4o_mini_index = create_index_by_path(gpt4o_mini_results)
gemini_index = create_index_by_path(gemini_results)

print("\nCalculando similaridades...")
results_comparison = []

for i, image_path in enumerate(baseline_index.keys()):
    print(f"\rProcessando {i+1}/166...", end="")

    baseline_data = baseline_index.get(image_path)
    gpt4o_mini_data = gpt4o_mini_index.get(image_path)
    gemini_data = gemini_index.get(image_path)

    baseline_text = extract_text_from_analysis(baseline_data)
    gpt4o_mini_text = extract_text_from_analysis(gpt4o_mini_data)
    gemini_text = extract_text_from_analysis(gemini_data)

    sim_gpt4o_mini = calculate_semantic_similarity(baseline_text, gpt4o_mini_text) if gpt4o_mini_text else 0
    sim_gemini = calculate_semantic_similarity(baseline_text, gemini_text) if gemini_text else 0

    results_comparison.append({
        'test_number': i + 1,
        'image_path': image_path,
        'similarity_gpt4o_mini': sim_gpt4o_mini * 100,
        'similarity_gemini': sim_gemini * 100,
        'baseline_time': baseline_data.get('response_time', 0) if baseline_data else 0,
        'gpt4o_mini_time': gpt4o_mini_data.get('response_time', 0) if gpt4o_mini_data else 0,
        'gemini_time': gemini_data.get('response_time', 0) if gemini_data else 0
    })

print("\n")

# Estatisticas
similarities_gpt4o = [r['similarity_gpt4o_mini'] for r in results_comparison]
similarities_gemini = [r['similarity_gemini'] for r in results_comparison]
times_baseline = [r['baseline_time'] for r in results_comparison if r['baseline_time'] > 0]
times_gpt4o = [r['gpt4o_mini_time'] for r in results_comparison if r['gpt4o_mini_time'] > 0]
times_gemini = [r['gemini_time'] for r in results_comparison if r['gemini_time'] > 0]

stats = {
    'gpt4o_mini': {
        'media_similaridade': np.mean(similarities_gpt4o),
        'std_similaridade': np.std(similarities_gpt4o),
        'media_tempo': np.mean(times_gpt4o) if times_gpt4o else 0,
    },
    'gemini': {
        'media_similaridade': np.mean(similarities_gemini),
        'std_similaridade': np.std(similarities_gemini),
        'media_tempo': np.mean(times_gemini) if times_gemini else 0,
    },
    'baseline': {
        'media_tempo': np.mean(times_baseline) if times_baseline else 0,
    }
}

print("="*60)
print("RESULTADOS")
print("="*60)
print(f"\nGPT-4o-mini vs Baseline: {stats['gpt4o_mini']['media_similaridade']:.2f}%")
print(f"Gemini vs Baseline: {stats['gemini']['media_similaridade']:.2f}%")
print(f"\nTempo medio - Baseline: {stats['baseline']['media_tempo']:.2f}s")
print(f"Tempo medio - GPT-4o: {stats['gpt4o_mini']['media_tempo']:.2f}s")
print(f"Tempo medio - Gemini: {stats['gemini']['media_tempo']:.2f}s")


def convert_to_native(obj):
    if isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_native(i) for i in obj]
    return obj


results_json = {
    'metadata': {
        'data_analise': datetime.now().isoformat(),
        'total_testes': 166,
        'metodo': 'Sentence Transformers (all-mpnet-base-v2)'
    },
    'estatisticas': convert_to_native(stats),
    'resultados_detalhados': convert_to_native(results_comparison)
}

output_path = r"C:\Users\Usuário\Desktop\benchmark_tests\data\processed\resultados_similaridade.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos em: {output_path}")
