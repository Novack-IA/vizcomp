# Weighted Loss no Ultralytics: como e por quê

Este documento explica linha a linha o que foi feito em `train.py` para implementar
uma função de perda ponderada por classe, usando o Ultralytics como framework de treino.

---

## Contexto: por que ponderação de classes?

O dataset SMART-OM é severamente desbalanceado:

| Classe       | Imagens | % do total |
|--------------|---------|------------|
| Normal       | 2145    | 86,9%      |
| Variation    | 179     | 7,2%       |
| OPMD         | 125     | 5,1%       |
| Oral Cancer  | 20      | 0,8%       |

Sem nenhuma correção, a rede aprende a prever "Normal" para quase tudo e ainda
assim obtém ~87% de acurácia. Isso é inútil clinicamente: queremos detectar as
classes raras (lesões), não só acertar a maioria.

A solução é **penalizar mais os erros nas classes raras durante o treino**,
o que é feito via pesos na função de perda.

---

## Parte 1 — Calculando os pesos (`compute_class_weights`, linhas 56–69)

```python
def compute_class_weights(data_dir: Path) -> list[float]:
    train_dir  = data_dir / "train"
    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    counts     = [len(list(d.iterdir())) for d in class_dirs]   # n_c por classe
    n_total    = sum(counts)                                      # N total
    n_cls      = len(counts)                                      # C classes
    weights    = [n_total / (n_cls * c) for c in counts]         # fórmula
    mean_w     = sum(weights) / len(weights)
    weights    = [w / mean_w for w in weights]                    # normaliza média=1
    return weights
```

### Fórmula aplicada

$$w_c = \frac{N}{C \cdot n_c}$$

Onde:
- $N$ = total de imagens de treino
- $C$ = número de classes
- $n_c$ = número de imagens da classe $c$

Essa é a **frequência inversa**: classes com menos amostras recebem pesos maiores.

**Normalização:** os pesos são divididos pela média para que a loss total tenha
a mesma escala que sem ponderação. Sem isso, o learning rate ficaria des-calibrado.

### Exemplo numérico (task `one_stage` com oversampling)

| Classe      | $n_c$  | $w_c$ bruto | $w_c$ normalizado |
|-------------|--------|-------------|-------------------|
| normal      | 1716   | 0,44        | ~0,27             |
| oral_cancer | 160    | 4,75        | ~2,93             |
| opmd        | 300    | 2,53        | ~1,56             |
| variation   | 300    | 2,53        | ~1,56             |

Um erro em `oral_cancer` vale ~11× mais que um erro em `normal` durante o treino.

---

## Parte 2 — A interface do Ultralytics com `criterion`

Antes de entender o que mudamos, é preciso entender como o Ultralytics chama a loss.

Internamente, `ClassificationTrainer` usa `v8ClassificationLoss`, que funciona assim:

```python
# Código interno do Ultralytics (simplificado)
class v8ClassificationLoss:
    def __call__(self, preds, batch):
        loss = F.cross_entropy(preds, batch["cls"], ...)
        return loss, loss.detach()
```

**Pontos importantes:**
1. `criterion` é um **objeto chamável** (não uma função pura).
2. Ele recebe `preds` (logits do modelo, shape `[N, C]`) e `batch` (dicionário).
3. `batch["cls"]` contém os labels verdadeiros (tensor de inteiros, shape `[N]`).
4. Retorna uma **tupla** `(loss, loss.detach())` — o Ultralytics espera exatamente isso.

O Ultralytics **não expõe** pesos de classe como argumento padrão.
Por isso precisamos substituir o `criterion` por um nosso.

---

## Parte 3 — `WeightedClassificationLoss` (linhas 72–85)

```python
class WeightedClassificationLoss:

    def __init__(self, weight: torch.Tensor, label_smoothing: float = 0.0):
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = self.loss_fn(preds, batch["cls"])
        return loss, loss.detach()
```

### Linha 80 — `nn.CrossEntropyLoss`

```python
self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
```

**Primitivo PyTorch:** [`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

Esta loss combina internamente `LogSoftmax` + `NLLLoss`. A fórmula sem peso é:

$$\mathcal{L} = -\sum_{i} \log\hat{p}_{i,y_i}$$

Com o parâmetro `weight`:

$$\mathcal{L} = -\sum_{i} w_{y_i} \cdot \log\hat{p}_{i,y_i}$$

Cada sample recebe um peso igual ao peso da sua **classe verdadeira** $y_i$.
Amostras de classes raras contribuem mais para o gradiente total.

O parâmetro `weight` deve ser um `torch.Tensor` com shape `[C]`,
onde `C` é o número de classes, **na mesma ordem que o modelo conhece as classes**.

`label_smoothing` é um regularizador que suaviza os targets duros (0/1) para
distribuições suaves (ε/(C-1) para classes erradas, 1-ε para a classe certa),
reduzindo overconfidence. No nosso caso, herdamos o valor do Ultralytics (padrão = 0).

### Linha 83 — desempacotando `preds`

```python
preds = preds[1] if isinstance(preds, (list, tuple)) else preds
```

Alguns modelos (especialmente com AuxHead ou múltiplas saídas) retornam
uma tupla `(aux_logits, main_logits)`. O índice `[1]` pega os logits principais.
Se `preds` já é um tensor, passa direto.

### Linha 84 — calculando a loss

```python
loss = self.loss_fn(preds, batch["cls"])
```

`batch["cls"]` é um tensor de inteiros `[N]` com o índice da classe verdadeira
de cada imagem no batch. A `CrossEntropyLoss` espera:
- `input`: logits `[N, C]` (antes do softmax — a loss aplica softmax internamente)
- `target`: índices `[N]` (dtype `torch.long`)

### Linha 85 — retorno em tupla

```python
return loss, loss.detach()
```

**Primitivo PyTorch:** [`Tensor.detach()`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

O Ultralytics espera `(loss_para_backward, loss_para_log)`.
- O primeiro é usado para `loss.backward()` — precisa ter gradiente.
- O segundo é para logging/métricas — não precisa de gradiente, `.detach()` corta o grafo computacional.

---

## Parte 4 — `WeightedCLSTrainer` (linhas 88–107)

```python
class WeightedCLSTrainer(ClassificationTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, class_weights=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._class_weights = class_weights

    def set_class_weights(self):
        if self._class_weights is None:
            return
        w = torch.tensor(self._class_weights, dtype=torch.float32).to(self.device)
        label_smoothing = getattr(self.args, "label_smoothing", 0.0)
        unwrap_model(self.model).criterion = WeightedClassificationLoss(w, label_smoothing)
        print(f"WeightedCrossEntropyLoss: {[f'{x:.3f}' for x in self._class_weights]}")
```

### Por que subclassear `ClassificationTrainer`?

O Ultralytics não permite trocar a loss via argumento. A única forma correta é
estender o trainer e usar os hooks do ciclo de vida de treino.

### Linha 97–99 — `__init__`

```python
def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, class_weights=None):
    super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
    self._class_weights = class_weights
```

O Ultralytics instancia o trainer como:

```python
trainer(overrides=args, _callbacks=callbacks)
```

Por isso o `__init__` precisa aceitar exatamente `cfg`, `overrides` e `_callbacks`
(mais o nosso `class_weights`). O `super().__init__` chama o `ClassificationTrainer`
original com os três argumentos padrão.

### Linha 101–107 — `set_class_weights()` (o hook)

```python
def set_class_weights(self):
    ...
    w = torch.tensor(self._class_weights, dtype=torch.float32).to(self.device)
    ...
    unwrap_model(self.model).criterion = WeightedClassificationLoss(w, label_smoothing)
```

`set_class_weights()` é um método do `ClassificationTrainer` original chamado
**dentro de `_setup_train()`**, depois que:
1. O modelo foi movido para o device (GPU/CPU).
2. O dataloader foi criado.
3. O otimizador foi configurado.

Isso garante que sabemos qual device usar quando criamos o tensor de pesos.

#### Linha 104 — movendo os pesos para o device correto

```python
w = torch.tensor(self._class_weights, dtype=torch.float32).to(self.device)
```

**Primitivos PyTorch:**
- [`torch.tensor()`](https://pytorch.org/docs/stable/generated/torch.tensor.html): cria um tensor a partir de uma lista Python.
- [`Tensor.to(device)`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html): move o tensor para o device do modelo.

Os pesos **precisam estar no mesmo device que os logits** durante o forward pass.
Se o modelo está na GPU e os pesos estão na CPU, o PyTorch lança um erro de device mismatch.

#### Linha 106 — `unwrap_model`

```python
unwrap_model(self.model).criterion = WeightedClassificationLoss(w, label_smoothing)
```

**`unwrap_model`** (de `ultralytics.utils.torch_utils`) desembrulha o modelo
caso ele esteja envolto em `DataParallel` ou `DistributedDataParallel`.
Em treino em múltiplas GPUs, o PyTorch envolve o módulo original num wrapper;
`unwrap_model` retorna o módulo raiz onde o `criterion` está de fato armazenado.

`.criterion = WeightedClassificationLoss(...)` simplesmente substitui o atributo
`criterion` do modelo, que é o objeto chamado durante o forward de treino.

---

## Parte 5 — Injeção via `functools.partial` (linha 127)

```python
TrainerCls = functools.partial(WeightedCLSTrainer, class_weights=class_weights)
```

**Primitivo Python:** [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial)

O problema: o Ultralytics chama o trainer assim:

```python
trainer(overrides=args, _callbacks=callbacks)   # sem class_weights!
```

Não é possível passar `class_weights` diretamente por `model.train()`.

`functools.partial` cria uma **versão pré-preenchida** de `WeightedCLSTrainer`
onde `class_weights` já está fixado. Quando o Ultralytics chama
`TrainerCls(overrides=args, _callbacks=callbacks)`, é equivalente a:

```python
WeightedCLSTrainer(overrides=args, _callbacks=callbacks, class_weights=class_weights)
```

O objeto `TrainerCls` se comporta como uma classe do ponto de vista do Ultralytics.

---

## Resumo do fluxo completo

```
model.train(trainer=TrainerCls, ...)
    │
    ├─ Ultralytics instancia: TrainerCls(overrides=..., _callbacks=...)
    │       └─ = WeightedCLSTrainer(..., class_weights=[0.27, 2.93, 1.56, 1.56])
    │
    ├─ _setup_train()
    │       └─ chama set_class_weights()
    │               └─ cria tensor w na GPU
    │               └─ substitui model.criterion por WeightedClassificationLoss(w)
    │
    └─ loop de treino (por batch)
            └─ preds = model(imgs)
            └─ loss, loss_log = model.criterion(preds, batch)
                    └─ = WeightedClassificationLoss.__call__(preds, batch)
                            └─ nn.CrossEntropyLoss(weight=w)(preds, batch["cls"])
```

---

## O que o Ultralytics faria sem nossa intervenção

```python
# v8ClassificationLoss original (Ultralytics)
loss = F.cross_entropy(preds, batch["cls"], label_smoothing=self.label_smoothing)
```

Sem `weight=`, toda classe tem peso 1.0: um erro em `oral_cancer` (20 amostras)
vale exatamente o mesmo que um erro em `normal` (2145 amostras). O modelo
aprende a ignorar as classes raras porque isso minimiza a loss de forma mais fácil.
