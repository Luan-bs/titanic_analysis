# Relatório de Otimização, *Overfitting Gap* e Comparação de Modelos

Neste relatório consolidamos os testes de variação de hiperparâmetros, a busca com `GridSearchCV` e o diagnóstico de **Overfitting Gap (`Train Accuracy − CV Accuracy`)** realizados nos notebooks `03_modeling.ipynb` e `04_tuning_comparison.ipynb`.

---

## 1. Diagnóstico Crítico: Por que avaliar o *Overfitting Gap* (`Treino vs CV`)?

Em bases compactas como a do Titanic (712 passageiros no `train.csv`), escolher um modelo olhando **apenas** para a média do Cross-Validation (`CV Accuracy`) pode induzir ao erro quando testamos centenas de combinações no `GridSearchCV`:
- Árvores muito profundas (**XGBoost com `max_depth=8`** e **Random Forest com `max_depth=12`**) alcançaram **84,55%** e **84,27%** no CV, mas quando medimos a acurácia no próprio conjunto de treino elas bateram **95,93%** e **93,12%** — abrindo um **Gap de Overfitting de +11,38% e +8,85%**! Na prática, as árvores decoraram micro-faixas de `Fare` e `Age` da base de treino.
- Já a **Logistic Regression** obteve alta acurácia no CV (**83,29%** com `C=0.5` e **82,73%** com `C=1.0`) mantendo praticamente a mesma nota no treino (**83,57%** e **83,43%**, um **Gap de apenas +0,28% a +0,70%**), além de entregar o maior **Recall (78,18%)** na validação.

| Modelo | Train Accuracy | CV Accuracy (5-Fold) | **Gap Overfitting** | CV F1-Score | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LR GridSearch (`C=0.5`)** | **0.8357** | 0.8329 | **+0.0028** | 0.7777 | 0.8182 | 0.7544 | **0.7818** | 0.7679 | 0.8674 |
| **LR Baseline (`C=1.0`)** | **0.8343** | 0.8273 | **+0.0070** | 0.7717 | 0.8182 | 0.7544 | **0.7818** | 0.7679 | 0.8630 |
| **RF Baseline (`max_depth=6`)** | 0.8581 | 0.8301 | +0.0280 | 0.7664 | 0.8182 | 0.7636 | 0.7636 | 0.7636 | 0.9012 |
| **XGB Baseline (`max_depth=3`)** | 0.8764 | 0.8343 | +0.0421 | 0.7700 | 0.8042 | 0.7455 | 0.7455 | 0.7455 | 0.8970 |
| **RF GridSearch (`max_depth=12`)** | 0.9312 | 0.8427 | *+0.0885* | 0.7847 | 0.8322 | 0.7925 | 0.7636 | 0.7778 | 0.8992 |
| **XGB GridSearch (`max_depth=8`)** | 0.9593 | **0.8455** | *+0.1138* | **0.7879** | **0.8462** | **0.8235** | 0.7636 | **0.7925** | **0.9056** |
| **RF (`max_depth=10, leaf=1`)** | 0.9621 | 0.8301 | *+0.1320* | 0.7685 | 0.8322 | 0.7925 | 0.7636 | 0.7778 | 0.8979 |

![Diagnóstico de Overfitting: Treino vs CV](./overfitting_gap_modelos.png)

![Comparação Baseline vs Tunados](./comparacao_baseline_vs_tuned.png)

---

## 2. Análise de Parâmetros: Random Forest

Variamos individualmente `max_depth`, `n_estimators` e `min_samples_leaf` medindo como cada parâmetro afeta o *overfitting*:

### 2.1. Impacto do `max_depth` (com `n_estimators=200`, `min_samples_leaf=1`)
Até `max_depth=6`, o `Gap Overfitting` fica controlado (`+0.0562`). A partir de `max_depth=8` até `None`, a acurácia no treino dispara para **98,46%**, enquanto o Cross-Validation cai para **80,34%** (um gap de **+18,12%** de *overfitting*):

| `max_depth` | Train Accuracy | CV Accuracy | **Gap Overfitting** | CV F1 | Val Accuracy | Val F1 | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3** | 0.8329 | 0.8244 | **+0.0085** | 0.7571 | 0.8252 | 0.7664 | 0.8968 |
| **4** | 0.8441 | 0.8300 | +0.0141 | 0.7664 | **0.8462** | **0.7963** | 0.9018 |
| **6** | 0.8919 | 0.8357 | +0.0562 | 0.7749 | 0.8322 | 0.7778 | **0.9064** |
| **8** | 0.9354 | **0.8371** | +0.0983 | **0.7784** | 0.8392 | 0.7850 | 0.8981 |
| **10** | 0.9621 | 0.8301 | +0.1320 | 0.7685 | 0.8322 | 0.7778 | 0.8979 |
| **12** | 0.9789 | 0.8244 | +0.1545 | 0.7634 | 0.8392 | 0.7850 | 0.8899 |
| **15** | 0.9846 | 0.8118 | +0.1728 | 0.7507 | 0.8252 | 0.7706 | 0.8913 |
| **None** | 0.9846 | 0.8034 | +0.1812 | 0.7412 | 0.8392 | 0.7965 | 0.8930 |

### 2.2. Impacto do `min_samples_leaf` (com `max_depth=10`, `n_estimators=200`)
Aumentar `min_samples_leaf` de `1` para `3` ou `4` poda as folhas que decoram passageiros isolados, reduzindo o `Gap Overfitting` pela metade (de `+0.1320` para `+0.0492` em `leaf=3`):

| `min_samples_leaf` | Train Accuracy | CV Accuracy | **Gap Overfitting** | CV F1 | Val Accuracy | Val F1 | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.9621 | 0.8301 | +0.1320 | 0.7685 | **0.8322** | **0.7778** | 0.8979 |
| **2** | 0.9101 | **0.8357** | +0.0744 | **0.7760** | **0.8322** | **0.7778** | 0.9021 |
| **3** | 0.8848 | **0.8357** | **+0.0491** | 0.7739 | 0.8252 | 0.7706 | **0.9045** |
| **4** | 0.8750 | 0.8315 | +0.0435 | 0.7694 | 0.8182 | 0.7593 | 0.9017 |
| **5** | 0.8652 | 0.8287 | +0.0365 | 0.7650 | 0.8112 | 0.7523 | 0.9037 |
| **8** | 0.8511 | 0.8300 | **+0.0211** | 0.7646 | 0.8182 | 0.7547 | 0.9035 |

![Curvas de Parâmetros - Random Forest](./rf_parametros.png)

---

## 3. Análise de Parâmetros: XGBoost

No XGBoost, o aumento de `max_depth` mostra claramente onde começa a memorização do treino:
- Em `max_depth=2` ou `3`, o `Gap Overfitting` é baixo (`+0.0225` a `+0.0421`).
- A partir de `max_depth=6`, `8` e `10`, a acurácia no treino atinge **95% a 97%**, enquanto o CV cai de `0.8357` para `0.8230`:

| `max_depth` | Train Accuracy | CV Accuracy | **Gap Overfitting** | CV F1 | Val Accuracy | Val F1 | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 0.8497 | 0.8273 | **+0.0224** | 0.7603 | 0.8042 | 0.7358 | 0.8969 |
| **3** | 0.8764 | 0.8343 | +0.0421 | 0.7700 | 0.8042 | 0.7455 | **0.8970** |
| **4** | 0.9045 | **0.8357** | +0.0688 | **0.7762** | 0.7902 | 0.7321 | 0.8925 |
| **5** | 0.9298 | 0.8272 | +0.1026 | 0.7654 | 0.8042 | 0.7500 | 0.8864 |
| **6** | 0.9537 | 0.8286 | +0.1251 | 0.7700 | **0.8182** | **0.7636** | 0.8835 |
| **8** | 0.9691 | 0.8230 | +0.1461 | 0.7626 | **0.8182** | **0.7636** | 0.8849 |
| **10** | 0.9733 | 0.8244 | +0.1489 | 0.7640 | **0.8182** | **0.7636** | 0.8855 |

![Curvas de Parâmetros - XGBoost](./xgb_parametros.png)

---

## 4. Análise de Parâmetros: Logistic Regression (Campeã em Estabilidade)

Na Regressão Logística, como o modelo aprende um hiperplano linear sobre os atributos padronizados (sem fatiar `Fare` e `Age` em dezenas de nós específicos), a curva de `Train Accuracy` anda colada na curva de `CV Accuracy` em todos os valores de `C`:

| `C` | Train Accuracy | CV Accuracy | **Gap Overfitting** | CV F1 | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | 0.7851 | 0.7725 | +0.0126 | 0.6247 | 0.7622 | 0.7561 | 0.5636 | 0.6458 | 0.8593 |
| **0.05** | 0.8258 | 0.8216 | +0.0042 | 0.7610 | 0.8042 | 0.7455 | 0.7455 | 0.7455 | **0.8671** |
| **0.10** | 0.8272 | 0.8244 | **+0.0028** | 0.7663 | 0.8112 | 0.7500 | 0.7636 | 0.7568 | 0.8651 |
| **0.25** | 0.8357 | 0.8272 | +0.0085 | 0.7696 | 0.8182 | 0.7544 | 0.7818 | 0.7679 | 0.8651 |
| **0.50** | **0.8357** | **0.8315** | **+0.0042** | **0.7770** | 0.8182 | 0.7544 | **0.7818** | 0.7679 | 0.8636 |
| **1.00** | 0.8343 | 0.8273 | +0.0070 | 0.7717 | 0.8182 | 0.7544 | **0.7818** | 0.7679 | 0.8630 |
| **2.00** | 0.8343 | 0.8244 | +0.0099 | 0.7685 | **0.8252** | **0.7679** | **0.7818** | **0.7748** | 0.8632 |
| **5.00** | 0.8357 | 0.8230 | +0.0127 | 0.7663 | 0.8182 | 0.7636 | 0.7636 | 0.7636 | 0.8626 |
| **10.0** | 0.8357 | 0.8202 | +0.0155 | 0.7636 | 0.8182 | 0.7636 | 0.7636 | 0.7636 | 0.8605 |

![Curvas de Parâmetros - Logistic Regression](./lr_parametros.png)

---

## 5. Curvas ROC e Matrizes de Confusão

![Curvas ROC e Matrizes de Confusão](./roc_e_matrizes_tuned.png)

---

## 6. Conclusão e Escolha Final para `data/inferred/test_inferred.csv`

Combinando **acurácia de validação cruzada (~83%)**, **maior Recall na validação (78,18%)** e **praticamente zero Overfitting Gap (< 0,7%)**, escolhemos a **Logistic Regression** como o modelo final do projeto. O arquivo `data/inferred/test_inferred.csv` foi gerado com as 179 predições (112 não sobreviventes e 67 sobreviventes, taxa de `37,43%`, alinhada à proporção real de ~38% do Titanic).
