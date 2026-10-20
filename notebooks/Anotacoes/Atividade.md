# KDD aplicado ao Titanic

Nesta atividade, vocês deverão desenvolver um processo completo de Knowledge Discovery in Databases (KDD) utilizando o conjunto de dados do Titanic.

O objetivo é aplicar, de maneira integrada, as etapas estudadas ao longo das aulas, desde a compreensão e preparação dos dados até a construção, interpretação e avaliação de modelos de Machine Learning.

---

## Dados Fornecidos

Serão disponibilizados:

- **`train.csv`**: conjunto de treinamento contendo a variável-alvo `Survived`;
- **`test.csv`**: conjunto sem a variável `Survived`, utilizado para a inferência final.

O conjunto de teste possui 179 passageiros cujos rótulos verdadeiros foram reservados pelo professor e serão utilizados posteriormente para avaliar a capacidade de generalização dos modelos.

---

## Desenvolvimento

O notebook deverá documentar todo o processo de KDD, incluindo:

### 1. Compreensão e Exploração dos Dados
- Análise das variáveis;
- Estatísticas descritivas;
- Análise de distribuições;
- Identificação de valores ausentes, *outliers* e possíveis inconsistências;
- Investigação das relações entre as características dos passageiros e a sobrevivência.

### 2. Pré-processamento
- Tratamento dos valores ausentes;
- Tratamento adequado das variáveis categóricas;
- Avaliação da necessidade de normalização ou padronização;
- Justificativa das decisões adotadas.

### 3. Feature Engineering
- Transformação e/ou criação de atributos que possam melhorar a representação dos dados;
- Avaliação da relevância das variáveis utilizadas.

### 4. Mineração de Dados
- Treinamento e comparação de mais de um algoritmo de classificação;
- Definição adequada da estratégia de validação;
- Escolha de métricas coerentes com o problema.

### 5. Avaliação e Interpretação
- Matriz de confusão;
- *Accuracy*, *Precision*, *Recall*, *F1-score* e outras métricas consideradas relevantes;
- Análise de erros;
- Interpretação do comportamento do modelo;
- Justificativa da escolha do modelo final.

### 6. Inferência
- Utilizar exclusivamente o modelo final para prever `Survived` no arquivo `test.csv`;
- Preservar exatamente a mesma quantidade e ordem das 179 observações do conjunto de teste.

---

## Entrega

Cada grupo deverá entregar:

1. **Notebook `.ipynb` executado**, contendo código, gráficos, resultados e interpretações;
2. **`test_inferred.csv`**, contendo as previsões de `Survived` para os 179 passageiros do conjunto de teste.

> **Atenção:** O notebook deve apresentar não apenas código, mas também justificativas e interpretações das decisões realizadas em cada etapa do KDD.

---

## Avaliação Final

Após a entrega, as previsões serão comparadas com um conjunto de respostas mantido exclusivamente pelo professor.

Além da qualidade da análise realizada no notebook, será comparado o desempenho real dos modelos sobre dados que os grupos nunca tiveram acesso aos rótulos.

---

> 🏆 **Ao final, será identificado o grupo que obteve a melhor capacidade de generalização no conjunto de teste.**
