# Proposta de Melhoria Metodológica — Validação, Comparação e Ablação

**Paper:** Classificação Multiclasse de Lesões da Mucosa Oral com Aprendizado Profundo em Imagens de Smartphone
**Escopo desta proposta:** corrigir o protocolo de validação para tornar as estimativas cientificamente defensáveis, e desenhar um estudo de ablação completo — **sem reexecutar ou modificar o código do benchmark de referência**.

---

## 1. Diagnóstico do problema de validação

O resultado central do paper — YOLO26m-cls superando o ResNet50 em +8,0 p.p. de sensibilidade macro na tarefa multiclasse, e o pipeline de dois estágios chegando a +10,26 p.p. — está medido sobre uma **partição única 80/20 dividida por imagem, sem estratificação por paciente**. As 2.469 imagens vêm de 331 pacientes (151 com lesões, 180 sem), e cada paciente contribui com até 8 regiões intraorais. Dividir por imagem permite que imagens do mesmo paciente caiam simultaneamente em treino e teste — vazamento de paciente (*patient leak*).

### Desfazendo o mal-entendido sobre o baseline

A premissa de que "não estratifiquei porque o paper de referência não estratifica, senão a comparação fica injusta" está parcialmente incorreta. O protocolo real do SMART-OM é:

1. Split **80/20 por imagem, fixo e sem estratificação por paciente** — é deste split que saem os números publicados que estão sendo comparados.
2. Validação cruzada 5-fold **usada apenas para tuning de hiperparâmetro**, dentro do conjunto de treino. Após o grid search, o modelo é re-treinado no treino inteiro sem split de validação.

Ou seja: o número final do benchmark **também** carrega patient leak. Os dois trabalhos estão no mesmo regime de vazamento, então a comparação atual **não é injusta** — mas ambos os números são igualmente frágeis perante um revisor de venue sério. O leak não prejudica a comparação; prejudica a credibilidade científica de ambos.

---

## 2. Estratégia de validação: dois níveis

Não escolher entre "comparar com eles (com leak)" e "fazer certo (sem comparabilidade)". Reportar **os dois níveis** e usar o gap entre eles como contribuição. Com acesso a H100, o custo de compute não é restrição.

### Nível 1 — Replicação fiel do protocolo do benchmark (com baseline)

Manter exatamente o split 80/20 por imagem já utilizado. São as Tabelas 3, 4 e 5 atuais, **incluindo a comparação com os números publicados de ResNet18/50**. Permanece no paper como **"comparação direta com o benchmark sob protocolo idêntico"**. Nada do trabalho atual é descartado, e nenhum código do benchmark precisa ser tocado — usa-se os valores reportados no paper deles.

### Nível 2 — Avaliação honesta com `StratifiedGroupKFold` (apenas modelos YOLO)

Onde entra o rigor:

- **Unidade de agrupamento = paciente.** O ID anonimizado está no nome do arquivo, no padrão `A_B_C.JPG`, em que `A` é o ID do paciente, `B` o local de coleta (R = Ranipet, W = camp World Vision) e `C` o código de duas letras da região intraoral.
- Usar `StratifiedGroupKFold` (scikit-learn) com **k=5**: balanceia a distribuição de classes entre folds *e* garante que nenhum paciente apareça em dois folds.
- Treinar **apenas os modelos YOLO** (v8m, v11m, 26m) k vezes, cada vez deixando 1 fold como teste; reportar **média ± desvio-padrão** sobre os 5 folds para cada métrica.
- O ResNet **não** é re-treinado neste nível, pois isso exigiria mexer no código do benchmark, o que está fora de escopo.

### Limitação honesta a declarar no texto

Como o ResNet não é avaliado sob o protocolo estratificado, **a comparação pareada "YOLO26m > ResNet" só é estatisticamente válida no Nível 1**. No Nível 2, a comparação passa a ser:

- **entre os próprios modelos YOLO** (qual variante generaliza melhor sob avaliação sem leak), e
- contra os números publicados do benchmark **apenas como referência informal/contextual** — explicitando que são regimes de avaliação diferentes e que a diferença não deve ser interpretada como comparação controlada.

Esta limitação deve ser declarada de forma direta. É preferível assumir o regime diferente a fingir uma comparação pareada que não existe.

### Enquadramento do resultado

O Nível 2 quase certamente produzirá números mais baixos que o Nível 1, porque o leak inflava o regime. Texto sugerido:

> A avaliação por split de imagem (Nível 1, idêntica ao benchmark) permite vazamento de paciente e infla as estimativas. Reportamos adicionalmente uma avaliação por validação cruzada estratificada por paciente (Nível 2) para os modelos propostos, fornecendo estimativas de generalização honestas. A queda observada entre os dois regimes quantifica o viés otimista induzido pelo protocolo por imagem. Como o baseline não foi reavaliado sob este protocolo, a comparação direta com o benchmark restringe-se ao Nível 1.

Com isso, o paper deixa de ser "mais um trabalho que rodou YOLO num dataset" e passa a "trabalho que mostrou que o protocolo por imagem é otimisticamente enviesado e mediu esse viés nos modelos propostos".

---

## 3. Significância estatística

Com 5 folds há 5 medidas por modelo no Nível 2. Para sustentar comparações **entre as variantes YOLO** (ex.: "YOLO26m > YOLOv8m em sensibilidade macro sob avaliação estratificada"):

- Aplicar **teste de Wilcoxon signed-rank pareado** (ou t-test pareado) sobre as sensibilidades macro por fold, entre os pares de modelos YOLO; reportar o p-valor.
- Com n=5 o poder estatístico é baixo. Recomenda-se **k=10** ou **5-fold repetido 3× com seeds distintas** (15 medidas), o que fortalece muito o argumento a custo de compute trivial em H100.

### Cuidado com a classe Câncer Oral

São apenas 20 imagens, concentradas em poucos pacientes. Sob `GroupKFold`, algum fold pode ficar com **zero** amostras de câncer no teste. Duas saídas:

1. Reportar a sensibilidade de câncer apenas nos folds em que a classe está presente no teste, sendo explícito sobre isso.
2. Para a tarefa de 4 classes, aceitar que câncer terá variância altíssima e declará-lo na seção de limitações — agora com base quantitativa (desvio entre folds), e não apenas qualitativa.

---

## 4. Estudo de ablação completo

Objetivo: **isolar a contribuição de cada componente do pipeline** e responder, com evidência, "de onde vem o ganho?". Hoje o paper aplica vários mecanismos simultaneamente (oversampling, loss ponderada, augmentation, pipeline de dois estágios) e reporta apenas o efeito agregado — um revisor competente vai pedir a decomposição.

Princípio metodológico para todas as ablações abaixo:

- Rodar sob o **Nível 2 (estratificado por paciente, k-fold)**, reportando **média ± desvio**, para que os efeitos não sejam artefatos de uma partição única.
- Mudar **um fator por vez** a partir de uma configuração de referência fixa.
- Fixar seed, hiperparâmetros e número de épocas entre as variantes de cada eixo de ablação.
- Usar **sensibilidade macro** como métrica primária de decisão (coerente com o resto do paper), reportando F1-macro e AUC como secundárias.

### 4.1. Ablação do tratamento de desbalanceamento (eixo principal)

Oversampling e loss ponderada por frequência inversa atacam o **mesmo** problema e podem ser parcialmente redundantes — ou até antagônicos (somados, podem super-corrigir e prejudicar a classe majoritária). Grade 2×2:

| Config | Oversampling | Loss ponderada |
|--------|:---:|:---:|
| A (baseline cru) | ✗ | ✗ |
| B | ✓ | ✗ |
| C | ✗ | ✓ |
| D (atual do paper) | ✓ | ✓ |

Interpretação esperada:

- **A** estabelece o piso sem nenhum tratamento — provavelmente colapsa nas classes raras (sensibilidade macro baixa, acurácia alta enganosa).
- **B vs. C** revela qual mecanismo carrega o ganho sozinho.
- **D vs. melhor de {B, C}** mede se combinar os dois ainda agrega ou se há redundância/saturação. Se D ≈ max(B, C), o paper pode **simplificar** para um único mecanismo — resultado limpo e citável.

Rodar a grade para o **melhor modelo** (YOLO26m-cls). Opcionalmente repetir para YOLOv8m-cls para checar se o padrão se mantém entre arquiteturas.

### 4.2. Ablação da arquitetura (já parcialmente no paper, formalizar)

O paper já mostra que o ganho não é uniforme na família YOLO (v8m e v11m ficam abaixo do ResNet50 no Nível 1). Formalizar como ablação explícita:

- YOLOv8m-cls vs. YOLOv11m-cls vs. YOLO26m-cls, **mesmo pipeline, mesmos folds**.
- Objetivo: atribuir o ganho às inovações arquiteturais do YOLO26 (arquitetura sem NMS, ProgLoss, otimizador MuSGD) e não a variação aleatória. O Wilcoxon pareado entre variantes (Seção 3) sustenta essa atribuição.

### 4.3. Ablação do pipeline de dois estágios

Decompor a contribuição do design em dois estágios:

- **One-stage** (classificador único de 4 classes) vs. **two-stage** (detector binário → classificador de lesões).
- Variar a escolha do modelo em cada estágio: testar se o ganho do two-stage depende de usar YOLO26m no Estágio 1 + YOLOv8m no Estágio 2, ou se é robusto a outras combinações.
- Reportar separadamente o desempenho **isolado do Estágio 2** (sobre as 760 imagens anômalas, sem erros propagados do Estágio 1) vs. o desempenho **end-to-end** — isso quantifica quanto erro o Estágio 1 injeta no pipeline.

### 4.4. Ablação de calibração (resolve a queda de AUC)

A AUC do pipeline caiu para 0,83 (vs. 0,92–0,93 dos one-stage) por falta de calibração entre estágios — as probabilidades dos dois estágios não estão em escala comum. Ablação:

- **Sem calibração** (atual) vs. **com calibração** das probabilidades de cada estágio antes da combinação.
- Métodos a comparar: *Platt scaling* (sigmoide) e *isotonic regression*.
- Reportar AUC macro e *Expected Calibration Error* (ECE) antes/depois. Espera-se recuperar boa parte da AUC sem perder sensibilidade — fechando o único flanco em que o pipeline perde hoje.

### 4.5. Ablação de data augmentation (opcional, se houver folga)

A augmentation atual mistura transformações geométricas (flip, rotação, translação, escala) com fotométricas (jitter de matiz/saturação/valor). Em imagens intraorais, **alterações de cor podem destruir o sinal diagnóstico** (eritroplakia, leucoplakia dependem de cor). Ablação sugerida:

- Augmentation completa (atual) vs. **só geométrica** vs. **sem jitter de matiz** (preservando matiz, que é o canal mais ligado ao diagnóstico).
- Objetivo: verificar se o jitter de cor está ajudando (regularização) ou atrapalhando (apagando sinal clínico). Resultado acionável para qualquer trabalho futuro no dataset.

### 4.6. Resumo da matriz de ablação

| Eixo | Fatores variados | Configs | Modelo(s) | Métrica de decisão |
|------|------------------|:---:|-----------|--------------------|
| 4.1 Desbalanceamento | oversampling × loss ponderada | 4 | YOLO26m (+ v8m opc.) | sens. macro |
| 4.2 Arquitetura | v8m / v11m / 26m | 3 | os três YOLO | sens. macro + Wilcoxon |
| 4.3 Pipeline | one vs. two-stage; combinações | ≥3 | YOLO | sens. macro end-to-end |
| 4.4 Calibração | nenhuma / Platt / isotonic | 3 | pipeline two-stage | AUC + ECE |
| 4.5 Augmentation | completa / geométrica / sem matiz | 3 | YOLO26m | sens. macro |

Priorização se o tempo apertar: **4.1 e 4.4 são as mais valiosas** (a primeira responde "de onde vem o ganho?", a segunda conserta a fraqueza de AUC). 4.2 já está meio feito. 4.3 e 4.5 são complementares.

---

## 5. Resumo acionável

1. Extrair o patient ID do nome de arquivo (`A` em `A_B_C.JPG`) e construir o vetor de grupos.
2. Manter a avaliação 80/20 atual como **Nível 1 — protocolo do benchmark**, preservando a comparação com os números publicados de ResNet.
3. Adicionar `StratifiedGroupKFold(k=5)` (ou repetido 3×) como **Nível 2**, treinando **apenas os modelos YOLO**; declarar explicitamente que a comparação pareada com o baseline se restringe ao Nível 1.
4. Reportar **média ± desvio** no Nível 2; rodar **Wilcoxon pareado** nas sensibilidades macro por fold entre as variantes YOLO.
5. Enquadrar o gap Nível 1 → Nível 2 como contribuição: medição do viés do protocolo por imagem nos modelos propostos.
6. Executar a **matriz de ablação** (Seção 4) sob o Nível 2, mudando um fator por vez, priorizando desbalanceamento (4.1) e calibração (4.4).

---

## 6. Melhorias complementares

- **Esclarecer o "conjunto de validação" da Seção 4.3:** o texto afirma não usar conjunto de validação separado, mas seleciona o melhor estágio "por sensibilidade macro em seus respectivos conjuntos de validação". A contradição precisa ser resolvida para não sugerir seleção de modelo no teste.
- **Grad-CAM:** em *medical imaging*, explicabilidade é praticamente obrigatória para credibilidade clínica. Já mencionado em trabalhos futuros — vale antecipar para esta versão, ao menos qualitativamente para a classe OPMD.
