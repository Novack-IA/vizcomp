# Metodologia

## 1. Dataset

Utilizamos o SMART-OM (*SMARTphone-based dataset of Oral Mucosa images*), composto por 2.469 imagens intraorais coletadas de 331 pacientes em condições clínicas reais, capturadas com smartphones Android e iOS.

As imagens estão distribuídas em quatro classes diagnósticas:

| Classe | N | % |
|---|---|---|
| Normal | 2.145 | 86,9% |
| Variation from Normal | 179 | 7,2% |
| OPMD | 125 | 5,1% |
| Oral Cancer (OC) | 20 | 0,8% |

Cada imagem corresponde a uma das oito regiões intraorais (língua dorsal, língua ventral, mucosa jugal esquerda, mucosa jugal direita, lábio superior, lábio inferior, arcada superior, arcada inferior). O pool completo de imagens é tratado como um único conjunto, sem separação por região, em conformidade com a metodologia do paper de referência.

Utilizamos exclusivamente as imagens da subpasta `01. Unannotated` de cada classe, que contêm as imagens originais sem sobreposição visual de anotações.

## 2. Divisão Treino/Teste

O dataset foi dividido aleatoriamente em 80% para treino e 20% para teste, sem estratificação por paciente ou por região intraoral, replicando o protocolo do paper de referência. Não foi utilizado conjunto de validação separado durante o treinamento final.

## 3. Modelos

Avaliamos a variante **medium (`-m`)** de três famílias de arquiteturas YOLO no modo classificação (`-cls`), todas treinadas via Ultralytics:

- **YOLOv8m-cls**
- **YOLOv11m-cls**
- **YOLO26m-cls**

Todos os modelos são inicializados com pesos pré-treinados no ImageNet e submetidos a fine-tuning no SMART-OM (*transfer learning*). A escolha dessas arquiteturas constitui a **primeira contribuição** do trabalho: enquanto o paper de referência avalia modelos CNN clássicos (ResNet18/34/50, VGG16, EfficientNet-b0) e um Vision Transformer (ViT), propomos arquiteturas unificadas e mais modernas otimizadas para inferência em tempo real.

## 4. Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| Epochs | 200 |
| Early stopping patience | 50 |
| Batch size | 32 |
| Image size | 320 px |
| Otimizador | AdamW (lr₀ = 1e-3, lrf = 0,01) |
| Flip horizontal | 0,5 |
| Rotação | ±10° |
| Translação | 10% |
| Escala | 30% |
| Jitter de matiz (hsv_h) | 0,005 |
| Jitter de saturação (hsv_s) | 0,7 |
| Jitter de valor (hsv_v) | 0,4 |

O jitter de matiz foi reduzido ao mínimo (0,005 vs. padrão 0,015) porque a cor das mucosas orais é clinicamente diagnóstica — alterações de tonalidade são sinal de lesão — e distorções agressivas de cor durante o treino poderiam prejudicar essa discriminação.

Não realizamos grid search de hiperparâmetros, diferindo do paper de referência que utilizou busca em grade com 5-fold cross-validation.

## 5. Tratamento do Desbalanceamento de Classes

Dado o severo desbalanceamento do dataset (Normal representa 86,9% das imagens), aplicamos **loss ponderada** durante o treinamento. Os pesos de cada classe são calculados pelo método de frequência inversa:

$$w_c = \frac{N}{C \cdot n_c}$$

onde $N$ é o total de amostras de treino, $C$ é o número de classes e $n_c$ é o número de amostras da classe $c$ no conjunto de treino.

A implementação no Ultralytics exigiu duas adaptações: (1) o `ClassificationTrainer` não expõe pesos de classe como parâmetro, sendo necessário subclassear e injetar a loss via hook interno; (2) o Ultralytics instancia o trainer com assinatura fixa `trainer(overrides, _callbacks)`, impossibilitando passar `class_weights` diretamente — resolvido com `functools.partial`.

Os pesos são normalizados pela média para que a escala da loss total não se altere em relação ao caso sem ponderação, mantendo o learning rate calibrado:

$$w_c \leftarrow \frac{w_c}{\bar{w}}$$

```python
import functools
import torch, torch.nn as nn
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils.torch_utils import unwrap_model

class WeightedClassificationLoss:
    """Replica a interface de v8ClassificationLoss com CrossEntropyLoss ponderada."""
    def __init__(self, weight, label_smoothing=0.0):
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = self.loss_fn(preds, batch["cls"])   # batch é dict no Ultralytics
        return loss, loss.detach()

class WeightedCLSTrainer(ClassificationTrainer):
    def __init__(self, cfg, overrides=None, _callbacks=None, class_weights=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._class_weights = class_weights

    def set_class_weights(self):
        """Hook do Ultralytics chamado após o modelo estar no device."""
        w = torch.tensor(self._class_weights, dtype=torch.float32).to(self.device)
        unwrap_model(self.model).criterion = WeightedClassificationLoss(
            w, label_smoothing=getattr(self.args, "label_smoothing", 0.0)
        )

# functools.partial preenche class_weights antes da instanciação pelo Ultralytics
TrainerCls = functools.partial(WeightedCLSTrainer, class_weights=class_weights)
model.train(trainer=TrainerCls, ...)
```

## 6. Tarefas de Classificação e Treinamentos

### 6.1 One-Stage: Classificação Multiclasse (4 classes)

Classificação direta nas quatro classes (Normal, Variation from Normal, OPMD, OC), replicando a tarefa multi-classe do paper de referência com as novas arquiteturas.

**Treinamentos:** 3 (um por família de modelo)

### 6.2 Two-Stage: Pipeline Binário + 3 classes

A **segunda contribuição** é a proposta e avaliação de um pipeline de classificação em dois estágios.

**Estágio 1 — Detector de anomalias (binário):**
Classifica cada imagem como `Normal` ou `Anormal` (Variation + OPMD + OC agrupados). A prioridade é **minimizar falsos negativos** (não deixar escapar casos anômalos), o que é clinicamente crítico. Os pesos de classe são calculados sobre as 2 classes binárias.

**Treinamentos:** 3 (um por família de modelo)

**Estágio 2 — Classificador de lesões (3 classes):**
Aplicado às imagens classificadas como Anormal pelo Estágio 1. Distingue entre Variation from Normal, OPMD e OC. Treinado exclusivamente sobre as 324 imagens anômalas (Variation + OPMD + OC).

**Treinamentos:** 3 (um por família de modelo)

**Total de treinamentos: 9**

### 6.3 Avaliação do Pipeline Two-Stage

Reportamos dois cenários de avaliação:

1. **End-to-end:** métricas finais de 4 classes computadas propagando os erros do Estágio 1 — imagens classificadas incorretamente como Normal pelo Estágio 1 nunca chegam ao Estágio 2 e são contabilizadas como erros.
2. **Estágio 2 isolado (upper bound):** métricas do classificador de 3 classes rodado sobre o subconjunto de teste anômalo completo, assumindo Estágio 1 perfeito. Permite isolar a capacidade discriminativa do Estágio 2.

## 7. Métricas de Avaliação

Para cada modelo e tarefa, reportamos as seguintes métricas no conjunto de teste, em conformidade com o paper de referência:

- **Acurácia global**
- **Precisão, Sensibilidade (Recall), Especificidade e F1-Score** — médias macro e micro
- **Métricas por classe** — Precisão, Sensibilidade, Especificidade e F1-Score individuais
- **AUC-ROC** — macro e por classe

No Estágio 1 do two-stage, destacamos especialmente a **Sensibilidade da classe Anormal** como métrica primária, dada a importância clínica de minimizar falsos negativos.

## 8. Diferenças em Relação ao Paper de Referência

| Aspecto | Paper de referência | Este trabalho |
|---|---|---|
| Arquiteturas | ResNet18/34/50, VGG16, EfficientNet-b0, ViT | YOLOv8m-cls, YOLOv11m-cls, YOLO26m-cls |
| Busca de hiperparâmetros | Grid search com 5-fold CV | Hiperparâmetros padrão Ultralytics |
| Pipeline | One-stage apenas | One-stage + Two-stage |
| Framework | PyTorch (custom) | Ultralytics |
| Número de treinamentos | 12 (6 modelos × 2 tarefas) | 9 (3 modelos × 3 tarefas) |
