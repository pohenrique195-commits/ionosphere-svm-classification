# ionosphere-svm-classification
# Classificação da Base Ionosphere com SVM

Projeto de Machine Learning desenvolvido em Python para classificação binária da base de dados Ionosphere, utilizando máquinas de vetores de suporte (*Support Vector Machines — SVM*).

O projeto compara diferentes configurações de SVM, avalia seu desempenho por meio de métricas de classificação e apresenta visualizações das fronteiras de decisão utilizando Análise de Componentes Principais (PCA).

## Objetivos

- Aplicar modelos SVM a um problema de classificação binária.
- Comparar kernels lineares e RBF (*Radial Basis Function*).
- Investigar o efeito do parâmetro de regularização `C`.
- Avaliar os modelos utilizando acurácia e matrizes de confusão.
- Visualizar os dados e as regiões de decisão em duas dimensões utilizando PCA.

## Base de dados

O projeto utiliza a base **Ionosphere**, disponibilizada pelo UCI Machine Learning Repository.

A base contém medições de sinais de radar utilizadas para distinguir retornos classificados como bons ou ruins.

As classes são representadas no projeto da seguinte forma:

- `1 — good`: sinal classificado como bom.
- `0 — bad`: sinal classificado como ruim.

A base é carregada diretamente pelo pacote `ucimlrepo`, sem necessidade de baixar o arquivo manualmente.

Fonte: [UCI Machine Learning Repository — Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere).

## Modelos utilizados

Foram implementadas e comparadas três configurações de SVM:

| Modelo | Kernel | Parâmetro C |
|---|---|---:|
| Modelo 1 | Linear | 0,1 |
| Modelo 2 | Linear
