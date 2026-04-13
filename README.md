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

---

## Estrutura do projeto

```
├── main.py                        # Etapa 1: geração dos datasets
├── config.py                      # Configurações centralizadas
├── pyproject.toml
├── data/
│   └── raw/
│       ├── Sentences_75Agree.txt  # Dataset original (baixar manualmente)
│       ├── train_full.csv         # Train completo após split
│       └── test.csv               # Test fixo (nunca augmentado)
└── src/
    ├── data/
    │   ├── loader.py              # Lê e faz split do Financial PhraseBank
    │   └── restrictor.py         # Aplica restrição de dados por cenário
    └── augmentation/
        ├── generator.py          # Geração de uma sentença via Ollama
        └── pipeline.py           # Orquestra modelos × cenários
```

**Saída gerada** em `data/generated/<modelo>/`:
- `train_augmented_<ratio>pct.csv` — dataset de treino completo (real + sintético)
- `train_restricted_<ratio>pct.csv` — apenas dados reais (baseline sem augmentation)
- `synthetic_only_<ratio>pct.csv` — apenas amostras sintéticas geradas
- `augmentation_summary.csv` — resumo de todos os cenários

---

## Instalação

**Pré-requisitos:** Python 3.13+, [uv](https://github.com/astral-sh/uv), [Ollama](https://ollama.com)

```bash
# Instalar dependências
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

**Etapa 1 — Geração dos datasets augmentados:**

```bash
python main.py
```

O Ollama deve estar rodando em background antes de executar. Se não estiver:

```bash
ollama serve
```

---

## Configuração

Todas as configurações ficam em `config.py`:

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `OLLAMA_MODELS` | Lista de modelos Ollama | llama3.1, mistral:7b, phi3.5 |
| `AUGMENTATION_RATIOS` | Cenários de augmentation | [0.10, 0.25, 0.50] |
| `TEST_SIZE` | Fração reservada para teste | 0.20 |
| `SEED` | Semente de reprodutibilidade | 42 |
| `TEMPERATURE` | Temperatura de geração | 0.8 |
| `DATASET_FILE` | Caminho do arquivo local | `data/raw/Sentences_75Agree.txt` |
