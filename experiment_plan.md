# Plano: Experimento de Identificação e Verificação com DMD

## Objetivo

Script CLI `run_experiment.py` que recebe templates `.pkl` extraídos, computa uma matriz de
scores e gera métricas de **identificação** (CMC) e **verificação** (ROC / TAR@FAR).

---

## 1. Estrutura de entradas

### Templates (queries e galeria)

Seguem a hierarquia extraída por `extract_templates.py`:

```
{DATASET}/templates/dmd
├── 000
│   ├── 00
│   │   └── sd258_000_11-00_latent_bad.pkl      ← query (latente)
│   └── 01
│       └── sd258_000_11-01_template_bad.pkl    ← galeria (rolada)
├── 001
│   ├── ...
```

O **subject ID** é o nome do diretório de nível 1 (`000`, `001`, ...).
O **sample ID** é o nome do diretório de nível 2 (`00`, `01`, ...).
O **dataset alias** é o primeiro token do nome do arquivo antes de `_` (ex: `sd258`).

### Diretórios de distratores (galeria extra)

Distratores são templates adicionais que aumentam o tamanho da galeria, simulando uma base
biométrica real. Eles **podem compartilhar o mesmo subject ID** com queries, pois a
identidade real é definida pela chave composta `(dataset_alias, subject_id)`.

```
{DATASET_EXTRA}/templates/dmd/...    ← dataset alias diferente → sem matches genuínos
```

---

## 2. Chave de identidade: `(dataset_alias, subject_id)`

A genuinidade de um par é determinada pela combinação:

```
identity_key = (dataset_alias, subject_id)
```

onde:
- `dataset_alias` = primeiro token do nome do arquivo (`sd258_000_...pkl` → `"sd258"`)
- `subject_id` = nome do diretório de nível 1 (`000`, `001`, ...)

Um par `(query_i, gallery_j)` é **genuíno** se e somente se:
1. `identity_key(query_i) == identity_key(gallery_j)`
2. `path(query_i) != path(gallery_j)` (não é o mesmo arquivo)

Consequência: distratores de `sd27` com subject `000` **não** são genuínos de queries de
`sd258` com subject `000`.

### Múltiplos genuínos

Quando um sujeito tem mais de uma amostra na galeria (e.g., sample 01 e 02), **emite
warning** listando o sujeito e as amostras em conflito, e considera **todos** como genuínos.
No CMC, o rank do sujeito é o **melhor rank** (posição mais baixa) entre todos os genuínos.

---

## 3. Reconstrução do template DMD a partir do `.pkl`

O `.pkl` armazena o formato plano. Para usar o matcher DMD, reconstituímos o template:

| Campo `.pkl`          | Campo DMD          | Transformação                               |
|-----------------------|--------------------|---------------------------------------------|
| `embeddings` (N, 768) | `feature` (N, 768) | direto (torch tensor float32)               |
| `mask` (N, 768)       | `mask` (N, 64)     | `mask[:, ::12]` — desfaz o `repeat×12`     |
| `minutiae` (N, 4) CCW | `mnt` (1, N, 3) CW | `(360 − angle_ccw) % 360` na coluna 2      |

> **Nota:** `mask[:, ::12]` funciona porque `mask_expanded = repeat(mask_raw, 12, axis=1)`,
> então a coluna `12k` contém exatamente `mask_raw[:, k]`.

---

## 4. Matriz de scores

```
score_matrix[i, j] = score(query_i, gallery_j)
```

- Shape: `(Q, G)` onde `G = galeria_principal + Σ distratores`
- Implementação: `dmd.DmdMatcher().identify(queries, gallery, device, batch_size)`
- Normalização padrão do DMD (`Normalize=True`, `N_mean=1327`) — sem alteração
- A matriz é salva em disco (`.npy`) para reaproveitamento sem re-executar o matching

---

## 5. Métricas

### 5a. Identificação (1:N)

Para cada query `i`, ordenar a galeria por score decrescente e localizar os genuínos.

- **Rank genuíno**: menor rank (posição 1-indexada) dentre todos os genuínos de `i`
- **CMC@K**: fração de queries com rank genuíno ≤ K; plotar K = 1…20
- **Rank-1**: CMC@1

### 5b. Verificação (1:1)

Extrair da score matrix:
- **Scores genuínos**: `score_matrix[i, j]` onde `genuine_mask[i, j] == True`
- **Scores impostores**: demais células (amostrar se a matriz for muito grande)

Métricas:
- **ROC curve** (TPR vs FPR)
- **TAR@FAR=0.1%** e **TAR@FAR=1%**
- **EER** (Equal Error Rate)

---

## 6. Interface CLI

```shell
python run_experiment.py \
  --queries-dir   {DATASET}/templates/dmd       \  # filtrado por --query-regex / --query-list
  --gallery-dir   {DATASET}/templates/dmd       \  # filtrado por --gallery-regex / --gallery-list
  --query-regex   ".*-00-.*"                    \  # filtro para isolar queries
  --gallery-regex ".*-01-.*"                    \  # filtro para isolar galeria
  --distractors   {DATASET_EXTRA}/templates/dmd \  # repetível, zero ou mais (sem filtro)
  --output-dir    results/sd258_exp1            \
  --device        cuda                          \
  --batch-size    256
```

**Validações obrigatórias antes do cálculo:**
- Se algum arquivo aparecer simultaneamente em queries e galeria → **erro**, abortar
- Se algum sujeito tiver múltiplos genuínos na galeria → **warning** com lista

### Saídas em `--output-dir`

```
results/sd258_exp1/
├── scores.npy           # (Q, G) float32
├── query_index.csv      # i, identity_key, path
├── gallery_index.csv    # j, identity_key, path
├── genuine_mask.npy     # (Q, G) bool
├── cmc.csv              # rank, cmc_value
├── roc.csv              # fpr, tpr, threshold
└── metrics.json         # rank1, eer, tar_at_far_0.1, tar_at_far_1.0
```

---

## 7. Decisões fechadas

| # | Questão | Decisão |
|---|---------|---------|
| 1 | Múltiplos genuínos na galeria | Warning + considera todos; CMC usa melhor rank |
| 2 | Arquivo na query E na galeria | **Erro** — abortar execução |
| 3 | Normalização de scores | Padrão do DMD (`Normalize=True`), sem ajuste extra |
| 4 | Chave de identidade para distratores | `(dataset_alias, subject_id)` — não apenas `subject_id` |
