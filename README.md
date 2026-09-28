# Titanic: comparacao de modelos

Repositorio de estudos de KDD e classificacao supervisionada para prever a sobrevivencia de passageiros do Titanic. A proposta e manter um protocolo comum e comparar modelos diferentes de forma justa.

## Modelos

- [Arvore de Decisao](models/decision_tree/README.md): linha de base atual.
- [Random Forest](models/random_forest/README.md): ensemble de arvores.
- [Gradient Boosting](models/gradient_boosting/README.md): experimento com Gradient Boosting, XGBoost ou LightGBM.
- [Template para novos modelos](models/template/README.md): estrutura para futuras comparacoes.

## Estrutura

```text
.
├── data/
│   ├── raw/                  # dataset original, nunca alterado
│   ├── interim/              # dados apos limpeza inicial
│   ├── processed/            # datasets prontos para analise
│   └── splits/               # indices fixos de treino, validacao e teste
├── models/                   # um diretorio e um KDD por modelo
├── notebooks/                # entendimento comum dos dados
├── reports/
│   ├── decision_tree/        # metricas e figuras da arvore
│   ├── random_forest/        # metricas e figuras da floresta
│   ├── gradient_boosting/    # metricas e figuras do boosting
│   └── comparison/           # comparacao consolidada
├── src/titanic/              # codigo compartilhado do projeto
├── pyproject.toml            # dependencias e configuracao do uv
└── uv.lock                  # versoes reproduziveis
```

## KDD comum

1. **Selecao do dominio:** classificacao binaria da sobrevivencia (`Survived`).
2. **Entendimento dos dados:** analisar tipos, ausencias, distribuicao do alvo, duplicidades e possiveis vazamentos.
3. **Pre-processamento:** tratar ausencias, codificar categorias e separar treino, validacao e teste sem vazamento.
4. **Transformacao:** comecar com `Pclass`, `Sex`, `Age`, `SibSp`, `Parch` e `Fare`; documentar cada feature adicional.
5. **Mineracao:** treinar cada modelo em sua pasta, mantendo seed e divisao comparaveis.
6. **Avaliacao:** registrar accuracy, precision, recall, F1, matriz de confusao e ROC-AUC em `reports/<modelo>/`.
7. **Interpretacao:** documentar importancia das features, erros relevantes e limitacoes.

## Ambiente com uv

O projeto usa `pyproject.toml` e `uv.lock` como fonte de dependencias. Nao e necessario manter um `requirements.txt` duplicado.

```bash
uv sync --dev
uv run jupyter lab
```

Para executar o treinamento da arvore:

```bash
uv run python models/decision_tree/train.py
```

Dependencias especificas devem ser adicionadas ao projeto, por exemplo:

```bash
uv add xgboost
uv add lightgbm
```

O notebook `notebooks/extract_data.ipynb` baixa o dataset uma vez e salva em `data/raw/titanic_raw.csv`. Nos notebooks de modelo, os dados podem ser carregados com:

```python
from utils import carregar_dados

df = carregar_dados()
```

O `.gitignore` exclui ambientes virtuais, datasets locais e resultados gerados.
