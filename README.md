# Data Augmentation with Ollama

Experimento de data augmentation para classificação de sentimento financeiro.
Avalia o impacto de dados sintéticos gerados por LLMs locais no desempenho de um modelo **TinyBERT** fine-tuned no dataset **Financial PhraseBank**.

---

## Visão geral

O experimento simula cenários de escassez de dados removendo uma fração do conjunto de treino e substituindo-a por amostras sintéticas geradas via Ollama, mantendo o tamanho total do dataset constante.

**3 modelos LLM** × **3 cenários** = 9 datasets de treino augmentados:

| Cenário | Dados reais | Dados sintéticos |
|---------|-------------|-----------------|
| 10%     | 90% de N    | 10% de N        |
| 25%     | 75% de N    | 25% de N        |
| 50%     | 50% de N    | 50% de N        |

**Modelos Ollama utilizados:**
- `llama3.1:latest`
- `mistral:7b`
- `phi3.5:latest`

Para cada cenário, o **TinyBERT** é fine-tuned em duas variantes:
- `restricted` — apenas dados reais, `(1 - ratio) * N` amostras (baseline sem augmentation)
- `augmented` — dados reais + sintéticos, mantendo o tamanho total `N`

Total: **3 LLMs × 3 ratios × 2 variantes = 18 runs de fine-tuning**, todas avaliadas no mesmo test set fixo.

---

## Estrutura do projeto

```
├── main.py                        # Etapa 1: geração dos datasets
├── train.py                       # Etapa 2: fine-tuning do TinyBERT
├── config.py                      # Configurações centralizadas
├── pyproject.toml
├── data/
│   ├── raw/
│   │   ├── Sentences_75Agree.txt  # Dataset original (baixar manualmente)
│   │   ├── train_full.csv         # Train completo após split
│   │   └── test.csv               # Test fixo (nunca augmentado)
│   └── generated/<modelo>/        # Saída da Etapa 1
├── models/<modelo>/<variant>/     # Pesos do TinyBERT treinado (Etapa 2)
├── results/
│   └── finetuning_results.csv     # Métricas de cada run
└── src/
    ├── data/
    │   ├── loader.py              # Lê e faz split do Financial PhraseBank
    │   └── restrictor.py          # Aplica restrição de dados por cenário
    ├── augmentation/
    │   ├── generator.py           # Geração de uma sentença via Ollama
    │   └── pipeline.py            # Orquestra modelos × cenários
    └── training/
        ├── dataset.py             # SentimentDataset (PyTorch)
        ├── finetune.py            # Loop de treino + avaliação do TinyBERT
        └── pipeline.py            # Orquestra todas as runs de fine-tuning
```

**Saída da Etapa 1** em `data/generated/<modelo>/`:
- `train_augmented_<ratio>pct.csv` — dataset de treino completo (real + sintético)
- `train_restricted_<ratio>pct.csv` — apenas dados reais (baseline sem augmentation)
- `synthetic_only_<ratio>pct.csv` — apenas amostras sintéticas geradas
- `augmentation_summary.csv` — resumo de todos os cenários

**Saída da Etapa 2**:
- `models/<modelo>/<variant>_<ratio>pct/model/` — TinyBERT fine-tuned + tokenizer
- `results/finetuning_results.csv` — métricas (accuracy, F1 macro/weighted, precision, recall) por run

---

## Instalação

**Pré-requisitos:** Python 3.13+, [uv](https://github.com/astral-sh/uv), [Ollama](https://ollama.com)

```bash
# Instalar dependências (inclui torch, transformers, accelerate)
uv pip install -e .

# Baixar os modelos no Ollama
ollama pull llama3.1:latest
ollama pull mistral:7b
ollama pull phi3.5:latest
```

---

## Dataset

Baixe o arquivo `Sentences_75Agree.txt` do [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) e coloque em:

```
data/raw/Sentences_75Agree.txt
```

O arquivo tem o formato `frase@sentimento` por linha, onde sentimento é `positive`, `neutral` ou `negative`.

---

## Como rodar

### Etapa 1 — Geração dos datasets augmentados

O Ollama deve estar rodando em background:

```bash
ollama serve
```

Em outro terminal:

```bash
python main.py
```

Isso lê o Financial PhraseBank, faz o split train/test, e para cada `(modelo × ratio)` gera os arquivos `train_restricted_*.csv`, `train_augmented_*.csv` e `synthetic_only_*.csv` em `data/generated/<modelo>/`.

### Etapa 2 — Fine-tuning do TinyBERT

Depois que a Etapa 1 estiver concluída:

```bash
python train.py
```

Isso itera sobre cada `(LLM × ratio × variant)` lendo os CSVs de `data/generated/`, fine-tuna o `huawei-noah/TinyBERT_General_4L_312D` e avalia no test set fixo (`data/raw/test.csv`). As métricas vão sendo gravadas incrementalmente em `results/finetuning_results.csv` (resistente a interrupções), e os pesos finais ficam em `models/<modelo>/<variant>_<ratio>pct/model/`.

GPU é usada automaticamente quando disponível (com FP16); caso contrário roda em CPU.

---

## Configuração

Todas as configurações ficam em `config.py`:

### Augmentation (Etapa 1)

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `OLLAMA_MODELS` | Lista de modelos Ollama | llama3.1, mistral:7b, phi3.5 |
| `AUGMENTATION_RATIOS` | Cenários de augmentation | [0.10, 0.25, 0.50] |
| `TEST_SIZE` | Fração reservada para teste | 0.20 |
| `SEED` | Semente de reprodutibilidade | 42 |
| `TEMPERATURE` | Temperatura de geração | 0.8 |
| `DATASET_FILE` | Caminho do arquivo local | `data/raw/Sentences_75Agree.txt` |

### Fine-tuning (Etapa 2)

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `TINYBERT_MODEL` | Checkpoint base do HuggingFace | `huawei-noah/TinyBERT_General_4L_312D` |
| `MAX_SEQ_LENGTH` | Comprimento máximo dos tokens | 128 |
| `NUM_EPOCHS` | Épocas por run | 4 |
| `TRAIN_BATCH_SIZE` / `EVAL_BATCH_SIZE` | Batch size | 32 / 64 |
| `LEARNING_RATE` | Taxa de aprendizado | 5e-5 |
| `WEIGHT_DECAY` | Decaimento de pesos | 0.01 |
| `WARMUP_RATIO` | Fração de warmup do scheduler | 0.1 |
| `TRAIN_VARIANTS` | Variantes treinadas | `("restricted", "augmented")` |
