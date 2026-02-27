# Script de Extração de Bases Usando DMD

Eu tenho o repositório `fingerprint-dmd`. Nele eu tenho encapsulado o modelo do DMD.

Existe inclusive uma pipeline de extração de bases usando o DMD, na biblioteca `grids` em `extraction/steps/dmd_step.py`.

O objetivo agora é criar um script organizado, sem muitas complicações e overengineering, que extrai uma base da seguinte forma.

Imagine um dataset de fingerprints, dividido hierarquicamente usando o ID do individuo primeiro e depois o ID da captura. Por exemplo, em bases com mais de 10000 individuos poderemos ter o individuo 00000 com amostra 00, 01, 02... Em alguns casos, a amostra 00 pode ser uma latente coletada em cena de crime e a amostra 01 pode ser uma coletada em laboratório. O nome do arquivo pode conter informações adicionais, que não seguem um padrão.

```
{DATASET}/images/orig
├── 000
│   ├── 00
│   │   └── {DATASETALIAS}_000_11-00_{EXTRAINFO}.png
│   └── 01
│       └── {DATASETALIAS}_000_11-01_{EXTRAINFO}.png
├── 001
│   ├── 00
│   │   └── {DATASETALIAS}_001_11-00_{EXTRAINFO}.png
│   └── 01
│       └── {DATASETALIAS}_001_11-01_{EXTRAINFO}.png
```

No final, queremos converter isso em templates pickle:

```{DATASET}/templates/dmd
├── 000
│   ├── 00
│   │   └── {DATASETALIAS}_000_11-00_{EXTRAINFO}.pkl
│   └── 01
│       └── {DATASETALIAS}_000_11-01_{EXTRAINFO}.pkl
├── 001
│   ├── 00
│   │   └── {DATASETALIAS}_001_11-00_{EXTRAINFO}.pkl
│   └── 01
│       └── {DATASETALIAS}_001_11-01_{EXTRAINFO}.pkl
```

Para minúcias e embeddings (que é o nosso caso até o momento), vamos ter os campos: `minutiae` e `embeddings`. As dimensões são: `minutiae` é um np.array de inteiros no formato (N, 4), com colunas representando x, y, angle e qualidade. Já `embeddings` é um np.array de floats no formato (N, D), onde D é a dimensão do embedding. O número de minutias N pode variar entre templates. No caso do DMD, tem também um campo da máscara [0, 1] aplicada sobre os embeddings, que é (N,D) também.

Sobre a interação: quero que ele seja um programa CLI, onde o usuário informa quais os arquivos processados:

```shell
--input-dir (pasta inteira)
# Camadas de filtragem
--filter-regex (regex para filtrar os arquivos a serem processados, por exemplo: ".*-00-.*" para pegar apenas as amostras 00)
--filter-list (arquivo de texto com uma lista de arquivos a serem processados, um por linha)
```

```shell
--output-dir (pasta onde os templates serão salvos, mantendo a hierarquia relativa a partir do input-dir)
```

Outra entrada, para avaliar diferentes tipos de extratores de minúcias vão se arquivos de minúcias que também seguem a mesma hierarquia do diretório com as imagens:

```shell
{DATASET}/minutiae/{EXTRACTOR}
├── 000
│   ├── 00
│   │   └── sd258_000_11-00_latent_bad.min
│   └── 01
│       └── sd258_000_11-01_template_bad.min
├── 001
│   ├── 00
│   │   └── sd258_001_11-00_latent_bad.min
│   └── 01
│       └── sd258_001_11-01_template_bad.min
```

O arquivo .min tem estrutura fixa:

```shell
Converte e compara arquivos de minúcias `.min`.

O padrão é:

1) Coordenadas
a. origem: (0,0) no canto superior esquerdo da imagem
b. eixos: x → direita, y → baixo
c. unidade: px

2) Ângulos
    a. eixo zero: +x (eixo x positivo)
    b. sentido positivo: anti-horário (90° apontando para o topo da imagem, 180° para a esquerda, 270° para baixo). Note que isso independe do sistema de coordenadas definido em 1b
    c. unidade: graus
    d. tipo: inteiro (sem casas decimais)
    e. precisão: 1 grau (vide item 2d), arredondamento para o inteiro mais próximo.
    f. intervalo: [0, 360).

3) Convenção da direção
    a. Bifurcação: Do centro do Y (interseção das linhas de crista) para o gap da bifurcação. Descendo da crista para o vale.
    b. Terminação: Do centro do Y (interseção dos vales) para o gap do encontro dos vales. Subindo do vale para o término da crista.

4) Arquivo de minúcias (.min)
    a. Header
    #MIN X Y ANGLE QUALITY [TYPE EXTRA...]
    b. Campos
    - X: coordenada x da minúcia (vide item 1)
    - Y: coordenada y da minúcia (vide item 1)
    - ANGLE: ângulo da minúcia (vide item 2)
    - QUALITY: qualidade da minúcia de 0 a 100 (inteiro)
    - [TYPE EXTRA...]: campos adicionais opcionais, como exemplo tipo de minúcia, ignorados para comparação
    c. Extensão: .min
```