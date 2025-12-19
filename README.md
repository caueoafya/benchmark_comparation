# Benchmark de Modelos 

Avaliação comparativa entre o modelo atual (WB Assist/GPT-4.1) e duas alternativas: GPT-4o-mini e Gemini 2.5 Flash Lite.

## Metodologia

Usamos **similaridade semântica** para comparar textos pelo significado, não por palavras exatas. O processo funciona assim:

1. Cada texto (resposta do modelo e gabarito) é convertido em um vetor de 768 números
2. Esses vetores capturam o "significado" do texto
3. Comparamos os vetores usando similaridade de cosseno

Assim, "AVC isquêmico" e "Acidente Vascular Cerebral do tipo isquêmico" são reconhecidos como equivalentes (~95% similaridade).

## Stack Técnico

| Biblioteca | Função |
|------------|--------|
| `sentence-transformers` | Modelo de IA local (`all-mpnet-base-v2`) que gera os embeddings. Roda na máquina, sem API externa |
| `scikit-learn` | Cálculo de similaridade de cosseno entre vetores |
| `pandas` | Leitura e manipulação dos dados (JSON de benchmark, planilhas de GT) |
| `openpyxl` | Parser para ler os arquivos Excel com os gabaritos médicos |
| `numpy` | Operações matemáticas com arrays e estatísticas |

O modelo de embeddings é baixado uma vez (~400MB) e roda localmente. Não há chamadas de API nem custos por execução.

## Métricas

Avaliamos 86 testes com Ground Truth validado por médicos, medindo 4 dimensões:

| Métrica | O que avalia | Peso |
|---------|--------------|------|
| Similaridade Geral | Alinhamento global da resposta com o GT | 20% |
| Diagnóstico Semântico | Se identificou a condição correta | 25% |
| Tipo de Imagem | Se reconheceu o tipo de exame (OCT, TC, etc) | 15% |
| Cobertura de Fatos | Quantas informações relevantes incluiu | 15% |

O diagnóstico direto (match exato de palavras) tem peso de 25%, totalizando 50% do score focado no diagnóstico.

## Pipeline de Análise

Os scripts estão numerados na ordem de execução:

```
01 → Similaridade com baseline (primeira exploração)
02 → Similaridade com GT (comparação real)
03 → Match direto de diagnóstico (muito rigoroso, descartado)
04 → Avaliação por fatos atômicos
05 → Combinação das métricas em score único
06 → Reescala para valores apresentáveis
```

Cada script gera um JSON em `data/processed/` que alimenta o próximo.

## Resultado

| Modelo | Score Final | Tempo Médio | Speedup |
|--------|-------------|-------------|---------|
| WB Assist (Baseline) | 67.9% | 9.57s | 1.0x |
| GPT-4o-mini | 64.1% | 6.24s | 1.5x |
| Gemini 2.5 Flash Lite | 67.4% | 4.34s | 2.2x |

**Conclusão:** O Gemini tem qualidade praticamente igual ao baseline (diferença de 0.5 pontos) e é 2.2x mais rápido. Em 1000 consultas/dia, isso representa ~1.5 horas de tempo de espera economizado.

## Estrutura

```
src/analysis/       Scripts de cálculo (6 arquivos)
data/raw/           JSON original do benchmark
data/ground_truth/  Gabaritos médicos (planilhas Excel)
data/processed/     Resultados intermediários
docs/               Documentação adicional
```

## Como Executar

```bash
# Instalar dependências
pip install sentence-transformers pandas openpyxl scikit-learn

# Rodar avaliação
cd src/analysis
python 05_multi_dimensional.py  # Avaliação principal
python 06_rescale_scores.py     # Gera scores finais
```

Na primeira execução, o modelo de embeddings será baixado automaticamente (~400MB).
