================================================================================
PROJETO: Aumento de Dados com Ollama para Classificação de Sentimentos Financeiros
================================================================================

VISÃO GERAL
-----------
Este projeto investiga o efeito do aumento de dados baseado em LLMs no
ajuste fino de um classificador TinyBERT para análise de sentimentos em
textos financeiros. Três LLMs locais (servidos via Ollama) são utilizados
para gerar amostras de treino sintéticas que substituem uma fração dos dados
reais. O classificador é ajustado sob diferentes razões de aumento e as
métricas resultantes são comparadas com baselines restritos (apenas dados
reais) e com o conjunto completo de dados.

Dataset: Financial PhraseBank (subconjunto Sentences_75Agree, 3 classes:
positivo, neutro, negativo). Divisão treino/teste: 80/20, seed=42.

LLMs avaliados: llama3.1:latest, mistral:7b, phi3.5:latest (todos servidos
localmente via Ollama em http://localhost:11434).

Razões de aumento: 10%, 25%, 50%, 75% (fração dos dados reais substituída por
dados sintéticos, mantendo o tamanho total do conjunto de treino constante).

Classificador: huawei-noah/TinyBERT_General_4L_312D, ajustado por 4 épocas
com lr=5e-5, batch_size=32, weight_decay=0.01, warmup_ratio=0.1,
max_seq_length=128.

================================================================================
ARQUIVOS DE CÓDIGO-FONTE
================================================================================

config.py
    Arquivo central de configuração. Define todos os parâmetros experimentais:
    nomes dos modelos Ollama, razões de aumento, configurações da divisão
    treino/teste, mapeamento de rótulos, caminhos de arquivos, hiperparâmetros
    do TinyBERT e configurações de geração do Ollama (temperature=0.8,
    max_tokens=120, max_retries=3).

main.py
    Ponto de entrada para a Etapa 1 (geração de dados). Carrega o arquivo
    bruto do Financial PhraseBank, realiza a divisão treino/teste e executa
    o pipeline completo de aumento para todas as combinações (LLM x razão).
    Salva as divisões brutas e todos os arquivos CSV gerados, depois imprime
    o resumo do aumento. Sem parâmetros de linha de comando; todas as
    configurações são lidas do config.py.
    Execução: python main.py

train.py
    Ponto de entrada para as Etapas 2 e 3. Executa o pipeline completo de
    ajuste fino do TinyBERT para todas as combinações (LLM x razão x variante)
    (Etapa 2) e em seguida computa e salva as estatísticas resumidas (Etapa 3).
    Requer que main.py tenha sido executado previamente para que data/raw/ e
    data/generated/ estejam populados. Os resultados são salvos
    incrementalmente; execuções interrompidas retomam a partir do último
    experimento concluído.
    Execução: python train.py

src/augmentation/generator.py
    Chama a API HTTP do Ollama para gerar uma única frase financeira sintética
    para uma determinada classe de sentimento e LLM. Gerencia tentativas e
    parsing das respostas.

src/augmentation/merge.py
    Utilitário para combinar dados reais restritos com amostras sintéticas em
    um único DataFrame aumentado.

src/augmentation/pipeline.py
    Orquestra o laço de aumento sobre todas as combinações (LLM x razão).
    Para cada cenário: (1) restringe o conjunto real de treino, (2) gera
    amostras sintéticas via API do Ollama e (3) salva três arquivos CSV por
    cenário (ver data/generated/ abaixo). Também salva augmentation_summary.csv.

src/data/loader.py
    Carrega e parseia o arquivo de texto bruto do Financial PhraseBank,
    realiza a divisão estratificada treino/teste e salva
    data/raw/train_full.csv e data/raw/test.csv.

src/data/restrictor.py
    Redução estratificada do conjunto de treino para (1 - razão) * N amostras,
    utilizada para produzir o baseline restrito (apenas dados reais) em cada
    razão.

src/training/dataset.py
    Wrapper PyTorch Dataset que tokeniza as amostras de texto para entrada
    no TinyBERT.

src/training/finetune.py
    Ajusta o TinyBERT em um DataFrame de treino fornecido e avalia no conjunto
    de teste. Reinicia os pesos do modelo para o estado pré-treinado original
    antes de cada execução, garantindo uma comparação justa. Retorna acurácia,
    F1 (macro e ponderado), precisão (macro), recall (macro) e loss.

src/training/pipeline.py
    Orquestra o ajuste fino para todas as combinações (LLM x razão x variante)
    mais o baseline com dados completos. Carrega o tokenizador e o modelo base
    uma única vez, reutilizando-os entre as execuções. Salva os resultados
    incrementalmente em results/finetuning_results.csv e grava os modelos
    treinados em models/.

src/analysis/stats.py
    Computa três estatísticas resumidas a partir de finetuning_results.csv:
      [1] Delta aumentado menos restrito por (LLM x razão) — indica se o
          aumento ajudou ou prejudicou em relação ao treino apenas com dados
          reais reduzidos.
      [2] Variação das métricas entre as razões por (LLM x variante) — reporta
          mínimo, máximo, amplitude e inclinação por ponto percentual de aumento.
      [3] Ranking geral de todas as configurações (LLM x variante x razão)
          ordenado por F1-macro.
    Salva os CSVs de saída em results/stats/.

================================================================================
DADOS
================================================================================

data/raw/Sentences_75Agree.txt
    Arquivo original do Financial PhraseBank. Cada linha tem o formato:
    "frase@rótulo", onde o rótulo é um de positive, neutral, negative.
    Contém apenas frases com pelo menos 75% de concordância entre anotadores.

data/raw/train_full.csv
    Divisão completa de treino (80% do dataset, ~2758 amostras) gerada por
    main.py. Utilizada como baseline com dados completos em train.py.

data/raw/test.csv
    Divisão de teste reservada (20% do dataset) gerada por main.py. Utilizada
    para avaliação em todas as execuções de ajuste fino.

data/generated/augmentation_summary.csv
    Tabela resumida de todos os cenários de aumento: LLM, razão, número de
    amostras reais, número de amostras sintéticas e tamanho total do dataset.

data/generated/<llm_slug>/train_augmented_<razão>pct.csv
    Conjunto de treino aumentado completo para um dado LLM e razão. Contém
    amostras reais (restritas) e sintéticas embaralhadas. O tamanho total é
    igual ao tamanho original do conjunto de treino. <llm_slug> é o nome do
    modelo com ":" substituído por "-" (ex.: llama3.1-latest, mistral-7b,
    phi3.5-latest). Razões disponíveis: 10pct, 25pct, 50pct, 75pct.

data/generated/<llm_slug>/train_restricted_<razão>pct.csv
    Conjunto de treino apenas com dados reais reduzido para (1 - razão) * N
    amostras. Utilizado como controle sem aumento para cada razão. Mesmas
    convenções de <llm_slug> e razão indicadas acima.

data/generated/<llm_slug>/synthetic_only_<razão>pct.csv
    Apenas as amostras sintéticas geradas pelo LLM para um dado modelo e razão.
    Útil para inspecionar a qualidade da geração independentemente do treino.

================================================================================
MODELOS TREINADOS
================================================================================

models/baseline/full/model/
    TinyBERT ajustado no conjunto de treino real completo (train_full.csv).
    Serve como baseline de limite superior. Gerado automaticamente ao rodar
    train.py. Contém model.safetensors, config.json, tokenizer.json,
    tokenizer_config.json e training_args.bin.


================================================================================
RESULTADOS
================================================================================

results/finetuning_results.csv
    Tabela mestre de resultados com uma linha por execução de ajuste fino.
    Colunas: llm, variant, ratio, train_size, loss, accuracy, f1_macro,
    f1_weighted, precision_macro, recall_macro, runtime, samples_per_second,
    steps_per_second, epoch.

results/stats/augmented_vs_restricted.csv
    Delta por (LLM x razão) entre as variantes aumentada e restrita para cada
    métrica (delta absoluto e variação percentual). Um accuracy_delta ou
    f1_macro_delta positivo indica que o aumento melhorou o desempenho em
    relação ao treino apenas com dados reais reduzidos.

results/stats/variation_across_ratios.csv
    Por (LLM x variante x métrica): mínimo, máximo, amplitude, valores na
    primeira e última razão, inclinação por ponto percentual e número de
    pontos de razão observados. Indica a sensibilidade de cada configuração
    à razão de aumento.

results/stats/ranking.csv
    Todas as configurações (LLM x variante x razão) ordenadas por f1_macro
    em ordem decrescente, incluindo train_size e todas as métricas de avaliação.

results/grafico_barras.png / results/grafico_barras.pdf
    Gráfico de barras comparando o desempenho dos modelos entre LLMs,
    variantes e razões.

results/grafico_linhas.png / results/grafico_linhas.pdf
    Gráfico de linhas mostrando a evolução das métricas de avaliação conforme
    a razão de aumento aumenta, por LLM e variante.

================================================================================
ORDEM DE EXECUÇÃO
================================================================================

Etapa 1 — Gerar os datasets aumentados:
    python main.py
    Pré-requisito: Ollama em execução localmente com os três modelos baixados
    (ollama pull llama3.1:latest; ollama pull mistral:7b; ollama pull phi3.5:latest).
    Saída: pastas data/raw/ e data/generated/ populadas.

Etapas 2 e 3 — Ajuste fino e avaliação:
    python train.py
    Pré-requisito: Etapa 1 concluída.
    Saída: results/finetuning_results.csv, results/stats/, models/.

================================================================================
DEPENDÊNCIAS
================================================================================

Versão do Python: ver .python-version
Gerenciador de pacotes: uv (ver pyproject.toml e uv.lock para lista completa)
Instalação: uv sync
Principais bibliotecas: transformers, torch, scikit-learn, pandas, tqdm, httpx
