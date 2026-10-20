# Anotações e Conclusões da EDA

## Visão Geral (`train.csv`)
- **Linhas:** 712
- **Colunas:** 11

---

## Dicionário de Variáveis e Observações Iniciais

- **Survived**
  - `0` = Não (61.66% no treino)
  - `1` = Sim (38.34% no treino)
  - *Nota:* Feature Target.

- **Pclass**
  - `1` = 1ª Classe (Alta/Elite) — 171 passageiros
  - `2` = 2ª Classe (Média) — 150 passageiros
  - `3` = 3ª Classe (Baixa/Operária) — 391 passageiros
  - *Nota:* Variável muito importante, pois define a localização no navio e a prioridade nos botes.

- **Name**
  - Todos os 712 valores são únicos.
  - *Ideia:* O nome completo em si não serve direto no modelo, mas conseguimos extrair o título social (`Mr.`, `Mrs.`, `Miss.`, `Master.`, etc.), que ajuda tanto na classificação quanto para preencher as idades vazias.

- **Sex**
  - `male` e `female` (sem valores nulos).
  - *Nota:* Precisamos converter de texto para `int64` (`male = 0`, `female = 1`) para calcular correlação e treinar os modelos.

- **Age**
  - Idade mínima: `0.42` anos | Mediana: `28.5` anos | Média: `29.8` anos | Máxima: `80` anos.
  - Possui **137 valores nulos (19.24%)**, sendo a maioria concentrada na 3ª classe (102 nulos na 3ª classe, 26 na 1ª e 9 na 2ª).

- **siblings**
  - Indica a quantidade de irmãos (*siblings*) ou cônjuges (*spouses*) a bordo (varia de `0` a `8`).
  - *Atenção:* No `train.csv` a coluna está escrita como `"siblings"`, mas no `test.csv` está como `"SibSp"`. Precisamos padronizar o nome no pré-processamento para não dar erro na inferência.

- **Parch**
  - Indica o número de pais (*parents*) ou filhos (*children*) a bordo (varia de `0` a `6`).

- **Ticket**
  - 571 valores únicos e 141 repetidos (pessoas da mesma família ou grupo compravam com o mesmo número de bilhete).

- **Fare**
  - Indica a tarifa ou o preço pago pela passagem (Libras Esterlinas).
  - *Inconsistência encontrada:* Vários valores acima de `513` (como `7925`, `21075`, `262375`) perderam o ponto decimal quando foram exportados (estavam multiplicados por `1000`). No Titanic real a passagem mais cara custou `512.33`. Dividindo os valores `> 513` por `1000`, a mediana ficou em `14.45` e a média caiu de `4380.21` para `31.82`.

- **Cabin**
  - Indica o número da cabine onde o passageiro ficou alojado.
  - **Formato:** Uma letra seguida de números (como `C85`, `C123` ou `E46`), onde a letra inicial identifica o Deck (`A, B, C, D, E, F, G, T`).
  - Possui **552 valores nulos (77.53%)**: faltam `20.5%` na 1ª classe (35 de 171), `90.0%` na 2ª classe (135 de 150) e `97.7%` na 3ª classe (382 de 391) — ou perderam os registros ou a maioria da 2ª e 3ª classe não tinha cabine numerada.

- **Embarked**
  - Indica o porto de embarque (apenas **2 valores nulos** no treino, ambos da 1ª classe):
    - **C** = Cherbourg (França)
    - **Q** = Queenstown (Irlanda, atual Cobh)
    - **S** = Southampton (Inglaterra — a maioria embarcou aqui)
  - *Nota:* Aplicar One-Hot Encoding. Como é uma variável nominal, se usarmos `S = 0, C = 1, Q = 2`, o modelo vai achar que `Q` vale o dobro de `C`.

---

## Análise de Outliers (Método IQR) e Ajuste do `Fare`

- **Age:** Os outliers são os idosos acima de `66.0` anos (apenas 6 registros / `0.84%`). São valores reais, não devemos remover.
- **Fare (Antes vs Depois da correção):**
  - **Errado:** Q3 era `56.93`, IQR `48.88`, Limite Superior `130.25` (114 outliers / `16.01%`), distorcendo completamente o boxplot.
  - **Corrigido (`/ 1000` para `> 513`):** Q1 = `7.90`, Mediana = `14.45`, Q3 = `31.00`, IQR = `23.10`, Limite Superior = `65.66` (92 outliers / `12.92%`).
  - Os outliers que continuam em `Fare` são legítimos porque muitos viajavam com bilhete conjunto de família ou em cabines de luxo da 1ª classe.
- **siblings e Parch:** Aparecem muitos outliers no IQR porque mais de 75% dos passageiros viajavam sem pais/filhos (`Parch = 0`), então qualquer valor acima disso já passa do limite do boxplot.

---

## Distribuição das Variáveis vs Sobrevivência

Analisando os gráficos de contagem e densidade separados por `Survived`:
- **Classe por sobrevivência:** Na 1ª classe, a quantidade de sobreviventes supera bastante a de mortos. Na 2ª classe fica quase empatado. Já na 3ª classe, a mortalidade foi muito alta (quase 300 mortos para menos de 100 sobreviventes).
- **Idade por sobrevivência:** Crianças de `0 a 8 anos` tiveram um pico claro de sobrevivência (prioridade nos botes). Entre `18 e 30 anos` a densidade de mortos é maior que a de sobreviventes, e acima de `65 anos` quase ninguém sobreviveu.
- **SibSp e Parch por sobrevivência:** Passageiros com `0` acompanhantes morreram em grande maioria. Quem tinha `1` ou `2` irmãos/cônjuges ou pais/filhos teve mais sobreviventes do que mortos. Já famílias muito grandes (`SibSp >= 3` ou `Parch >= 4`) quase não sobreviveram.
- **Tarifa por sobrevivência:** Quem pagou passagens muito baratas (concentradas perto de `0 a 20`) morreu muito mais. A partir de tarifas acima de `30`, a densidade de sobreviventes fica sempre acima da de não-sobreviventes.

---

## Insights e Testes de Feature Engineering na EDA

1. **Tamanho da Família (`family_size`), Sozinho (`is_alone`) e Tarifa por Pessoa (`fare_per_person`):**
   - Criamos `family_size = siblings + Parch + 1`, `is_alone = (family_size == 1)` e `fare_per_person = Fare / family_size`.
   - O gráfico de barras mostrou que quem viajava sozinho (`family_size = 1`) teve ~30% de sobrevivência (`is_alone` teve correlação de `-0.22`), famílias de `2 a 4` membros tiveram as maiores taxas de sobrevivência (acima de 50% a 70%), e famílias com `5+` pessoas caíram drasticamente.
   - Dividir o `Fare` pelo tamanho da família reduziu o limite de outlier de `65.66` para `36.44` e deixou o valor da passagem por pessoa bem mais próximo da realidade de cada classe.

2. **Correlação de Pearson (Após corrigir `Fare`):**
   - `Sex`: `+0.55` (maior correlação positiva — mulheres sobreviveram muito mais).
   - `Pclass`: `-0.35` (correlação negativa moderada — quanto maior o número da classe, menor a sobrevivência).
   - `Fare` (`+0.28`) e `fare_per_person` (`+0.22`): antes do ajuste do `Fare` a correlação era quase zero (`0.07`), e depois da correção passou a mostrar relação real com a sobrevivência.
   - `is_alone`: `-0.22` (viajar sozinho diminui a chance de sobreviver).
   - `Embarked_C` (`+0.18`) vs `Embarked_S` (`-0.19`): quem embarcou na França sobreviveu mais (maior proporção de 1ª classe).

3. **Decks (`Cabin`) e Distribuição por Classe:**
   - Extraindo a primeira letra da cabine (`Deck`), vimos que `Deck_Unknown` (sem cabine) tem correlação `-0.35` com `Survived`, enquanto os decks `D` (`+0.17`), `B` (`+0.16`), `E` (`+0.16`) e `C` (`+0.15`) têm correlação positiva.
   - No heatmap de **Distribuição dos Decks por Classe**, ficou claro o motivo:
     - **1ª Classe:** Concentra os decks `C` (`24.0%`), `B` (`19.9%`), `D` (`13.5%`), `E` (`13.5%`), `A` (`8.2%`) e `T` (`0.6%`), com apenas `20.5%` de `Unknown`. Os decks `A`, `B`, `C` e `T` são exclusivos da 1ª classe.
     - **2ª Classe:** Tem `90.0%` como `Unknown`, e o restante apenas nos decks `F` (`5.3%`), `E` (`2.7%`) e `D` (`2.0%`).
     - **3ª Classe:** Tem `97.7%` como `Unknown`, aparecendo só nos decks mais baixos `G` (`1.0%`), `F` (`0.8%`) e `E` (`0.5%`).
   - Ou seja, criar uma coluna binária `has_cabin` (tem cabine ou não) ou usar o `Deck` captura muito bem a localização e o nível socioeconômico.

4. **Extração do Título a partir de `Name` (`Title`):**
   - Extraímos o título antes do ponto (ex: `Mr.`, `Miss.`, `Mrs.`, `Master.`) e agrupamos os equivalentes (`Mlle` e `Ms` em `Miss`, `Mme` em `Mrs`) e os títulos raros (`Dr`, `Rev`, `Col`, `Major`, `Lady`, `Sir`, `Don`, `Jonkheer`) em `Rare`.
   - **Taxas de sobrevivência e Correlação:**
     - `Mr` (412 passageiros): apenas `15.3%` sobreviveram (correlação forte de **`-0.56`**).
     - `Mrs` (107 passageiros): `78.5%` sobreviveram (correlação de **`+0.35`**).
     - `Miss` (144 passageiros): `70.8%` sobreviveram (correlação de **`+0.34`**).
     - `Master` (31 passageiros - meninos crianças): `54.8%` sobreviveram, muito acima dos homens adultos (`Mr`).
   - **Utilidade para preencher `Age`:** A mediana de idade muda muito conforme o título (`Master = 3.0 anos`, `Miss = 22.0 anos`, `Mr = 30.0 anos`, `Mrs = 35.0 anos`, `Rare = 49.0 anos`). Usar a mediana agrupada por `Title` (e `Pclass`) no pré-processamento vai ser muito mais preciso do que jogar a mediana geral de `28.5` anos para todo mundo.

---

## Próximos Passos (`02_preprocess.ipynb`)
1. Padronizar o nome da coluna `siblings` / `SibSp` no treino e no teste.
2. Aplicar a correção da escala de `Fare` (`> 513` dividido por `1000`) no `train.csv` e no `test.csv`.
3. Criar as features `Title`, `family_size`, `is_alone`, `fare_per_person` e `Deck` / `has_cabin`.
4. Tratar os valores ausentes:
   - `Embarked` (2 nulos no treino): preencher com a moda (`S`) ou pelo perfil da tarifa/classe.
   - `Age` (137 nulos no treino e 40 no teste): preencher com a mediana agrupada por `Pclass` e `Title` (calculada no treino para evitar vazamento de dados).
5. Codificar variáveis categóricas (`Sex` para `0/1` e One-Hot Encoding para `Embarked`, `Title` e `Deck`) garantindo que `train` e `test` fiquem com as mesmas colunas.
