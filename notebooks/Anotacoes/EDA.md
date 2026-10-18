# Anotações e Conclusões da EDA

## Visão Geral (`train.csv`)
- **Linhas:** 712
- **Colunas:** 11

---

## Dicionário de Variáveis

- **Survived**
  - `0` = Não
  - `1` = Sim
  - *Nota:* Feature Target 

- **Pclass**
  - `1` = 1ª Classe (Alta/Elite)
  - `2` = 2ª Classe (Média)
  - `3` = 3ª Classe (Baixa/Operária)
  - *Nota:* Variável parece importante, pois define os lugares do navio onde ficaram.

- **Name**

- **Sex**
  - Masculino e Feminino

- **Age**
  - Idade máxima: 80 anos
  - Idade mínima: 0.42 anos

- **siblings**
  - Indica a quantidade de irmãos (*siblings*) ou cônjuges (*spouses*) que o passageiro tinha a bordo do navio.
  - *Nota:* No `test.csv` está como `"SibSp"`, podemos mudar.

- **Parch**
  - Indica o número de pais (*parents*) ou filhos (*children*) que o passageiro tinha a bordo do Titanic.

- **Ticket**

- **Fare**
- Indica a tarifa ou o preço pago pela passagem (Libras Esterlinas).

- **Cabin**
  - Indica o número da cabine ou quarto onde o passageiro ficou alojado.
  - **Formato:** Uma letra seguida de números (como C85, C123 ou E46). A letra inicial identifica o Deck (conveses) onde ficava a cabine (Decks A, B, C, D, E, F, G).
  - *Ideia:* Pode servir para preencher Pclass vazios. Ex: Analisamos as cabines de 1ª classe e preenchemos como 1 em Pclass se a cabine for de 1ª classe.

- **Embarked**
  - Indica o porto de embarque, ou seja, a cidade onde o passageiro entrou no Titanic. Representado por três iniciais:
    - **C** = Cherbourg (França)
    - **Q** = Queenstown (Irlanda, atual Cobh)
    - **S** = Southampton (Inglaterra)

---

## Insights e Feature Engineering 
- **Tamanho da Família:** Podemos inferir a informação de `tamanho de família = siblings + Parch`.

---

## Próximos Passos
- **Analisar correlação:** `Survived` vs `Age`, `Sex`, `Cabin`, `Pclass`.
 