# Árvore de Decisão Simples

## KDD deste experimento

### 1. Seleção do domínio

Prever a sobrevivência de passageiros do Titanic a partir de informações pessoais e da viagem.

### 2. Seleção e entendimento dos dados

Usar o dataset Titanic com `Survived` como alvo. Conferir tipos, valores ausentes, distribuição da classe e possíveis vazamentos de informação.

### 3. Pré-processamento

Tratar `Age` e demais ausências, codificar `Sex`, selecionar as features comuns e separar treino e teste com `random_state` fixo.

### 4. Transformação

Começar com `Pclass`, `Sex`, `Age`, `SibSp`, `Parch` e `Fare`. Registrar qualquer feature criada, como tamanho da família ou título do passageiro.

### 5. Mineração

Treinar `DecisionTreeClassifier`. Comparar uma configuração base com ajustes de profundidade e critérios de divisão.

### 6. Avaliação e interpretação

Registrar acurácia, precisão, recall, F1, matriz de confusão e a árvore final. Comparar com os demais modelos usando exatamente o mesmo conjunto de teste.

## Arquivos esperados

- `train.py`: treinamento reproduzível.
- `notebook.ipynb`: exploração e visualizações do modelo.
- `README.md`: hipótese, decisões e resultados.