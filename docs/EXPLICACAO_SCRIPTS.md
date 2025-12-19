# Scripts de Avaliação - Explicação de Negócio

## Contexto

Pergunta de negócio: **"Podemos trocar o modelo atual (WB Assist/GPT-4.1) pelo Gemini 2.5 Flash Lite sem perder qualidade?"**

---

## 1. 01_semantic_similarity.py

**O que faz:** Compara GPT-4o-mini e Gemini com o baseline (similaridade semântica)

**Pergunta:** "O Gemini responde parecido com o modelo atual?"

**Limitação:** Mede similaridade com baseline, não com resposta correta

---

## 2. 02_similarity_vs_gt.py

**O que faz:** Compara todos os modelos com o Ground Truth (gabarito médico)

**Pergunta:** "Qual modelo responde mais parecido com o esperado pelos médicos?"

---

## 3. 03_direct_match.py

**O que faz:** Verifica se o diagnóstico do GT aparece literalmente na resposta

**Pergunta:** "O modelo acerta o diagnóstico usando as mesmas palavras?"

**Limitação:** Muito rigoroso - "AVC" vs "Acidente Vascular Cerebral" conta como erro

---

## 4. 04_atomic_facts.py

**O que faz:** Quebra respostas em fatos individuais e mede cobertura

**Pergunta:** "Quantas informações importantes o modelo menciona?"

---

## 5. 05_multi_dimensional.py

**O que faz:** Combina 4 métricas em score ponderado:
- Similaridade Geral (20%)
- Diagnóstico Semântico (25%)
- Diagnóstico Direto (25%)
- Tipo de Imagem (15%)
- Cobertura de Fatos (15%)

**Pergunta:** "Qual é a qualidade geral de cada modelo?"

---

## 6. 06_rescale_scores.py

**O que faz:** Ajusta scores para escala apresentável (0-100% intuitivo)

**Pergunta:** "Como apresentar os resultados para stakeholders?"

---

## Resultado Final

| Modelo | Score | Tempo | Speedup |
|--------|-------|-------|---------|
| Baseline | 67.9% | 9.57s | 1.0x |
| GPT-4o-mini | 64.1% | 6.24s | 1.5x |
| **Gemini** | **67.4%** | **4.34s** | **2.2x** |

**Resposta:** Sim, Gemini pode substituir o baseline (qualidade equivalente, 2.2x mais rápido)
