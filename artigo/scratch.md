# Semelhança entre Partidas de Dota 2 usando Autoencoders e Clustering


## Resumo
* Abstract. Este trabalho propõe a aplicação de autoencoder, juntamente com um classificador, para detectar semelhança entre partidas profissionais de Dota 2, um jogo eletrônico do gênero MOBA. A análise baseia-se em características como a composição dos heróis, desempenho econômico, tempo de partida e objetivos alcançados. O estudo é focado em analisar modelos, treinar em diferentes versões e comparar partidas de campeonatos. O modelo usa embedding para redução de dimensões em heróis escolhidos e banidos.
* Resumo. This paper proposes the application of a Multilayer Perceptron (MLP) neural network model to classify strategies used by professional teams in Dota 2 matches, a popular multiplayer online battle arena (MOBA) game. The analysis is based on features such as hero composition, economic performance, match duration, and in-game objectives. Using the “Dota 2 Pro League Matches 2016–2025” dataset from Kaggle, the model is trained to identify strategic patterns adopted by teams throughout the game. This study aims to contribute to automated methods for strategic analysis in the competitive eSports scene.


## 1 Introdução
* Assuntos
    * Dinheiro
    * Dados ricos,
    * Aplicação matemática e estatística
    * Estudo social e comportamental
    * Sponsors de análise de partida
* Definição do problema
    * Análise probabilística durante o jogo
    * Previsões de vitória, pick/bans
    * Estatísticas built-in com Dota+
    * Histórico, relatórios e estatísticas ao vivo
    * Narradores e Analistas de jogos ao vivo
    * Similaridades de jogos de torneios profissionais

* Solução
    * Procurar características relevantes
    * Pré processar o dataset bruto
    * Uso de redução de dimensionalidade dos dados
    * Agrupar jogos similares e comparar a acurácia usando aprendizado não supervisionado
    * Teste e análise supervisionada para jogos profissionais oficiais (The International)
## 2 Metodologia
* Diminuição de características
    * Autoencoders e usando espaço latente
        * VAE (?)
    * Clustering para agrupar os jogos semelhantes
        * Comparação entre k-means e OPTICS 
    * Análise supervisionada de jogos de The International
        * Dados de 2020 a 2024
        * Heróis escolhidos, proibidos
        * Itens
        * GPM e XPM
        * Duração
        * Região
        * KDA
        * Observer / Sentry
        * Objetivos
### 2.1 Conjunto de Dados
* Open Dota
* Kaggle
* Tiers
* Características Tabular
* Características Temporal
* Pré-processamento
    * Limpeza dos dados
    * Extração das características
    * Quais dados foram usados
* Normalização
    * Embedding de heróis e itens
### 2.2 Autoencoders
* Tipo de Autoencoders
    * Padrão
    * VAE
    * SAE
    * Deep Embedded Clustering (DEC)****
        * Minimizing the Kullback-Leibler (KL) divergence between a data distribution and an embedded distribution has been used for data visualization and dimensionality reduction (van der Maaten & Hinton, 2008). T-SNE, for instance, is a non-parametric algorithm in this school and a parametric variant of t-SNE (van der Maaten, 2009) uses deep neural network to parametrize the embedding. The complexity of t-SNE is O(n 2 ), where n is the number of data points, but it can be approximated in O(n log n) (van Der Maaten, 2014).
        * The proposed algorithm (DEC) clusters data by simultaneously learning a set of k cluster centers {µj ∈ Z} k j=1 in the feature space Z and the parameters θ of the DNN that maps data points into Z. DEC has two phases: (1) parameter initialization with a deep autoencoder (Vincent et al., 2010) and (2) parameter optimization (i.e., clustering), where we iterate between computing an auxiliary target distribution  and minimizing the Kullback–Leibler (KL) divergence to it. We start by describing phase (2) parameter optimization/clustering, given an initial estimate of θ and {µj} k j=1.
        * We initialize DEC with a stacked autoencoder (SAE) because recent research has shown that they consistently produce semantically meaningful and well-separated representations on real-world datasets
        * We use the standard unsupervised evaluation metric and protocols for evaluations and comparisons to other algorithms (Yang et al., 2010). For all algorithms we set the number of clusters to the number of ground-truth categories and evaluate performance with unsupervised clustering accuracy (ACC ): ACC = max  m  Pn  i=1 1{li = m(ci)}  n  , (10)  where li  is the ground-truth label, ci is the cluster assignment produced by the algorithm, and m ranges over all possible one-to-one mappings between clusters and labels.
    * Disentangled Variational Autoencoder based Multi-Label Classification with Covariance-Aware Multivariate Probit Model
* Cálculo de diferença usando Kullback–Leibler (KL) divergence
* Autoencoder com clustering
### 2.3 Clustering
* DEC precisa de número de cluster definido
    * Semi-supervisionado para pré treinamento dos pesos
        * Label de jogos, trabalho manual
    * Usado MINIST etc. onde já existe o número
* k-means eGaussian Mixture Models (GMM)
    * Quantidade de cluster não definida
* HDBSCAN e OPTICS
    * Cálculo do Epsilon baseado na distâncias de vizinhos
        * k-dist plot
* Spectral Clustering
    * Não linear, espaço não euclidiano
    * Dificultar visualização de dados
* Combining spectral clustering and embedding has been explored in Yang et al. (2010); Nie et al. (2011). Tian et al. (2014) proposes an algorithm based on spectral clustering, but replaces eigenvalue decomposition with deep autoencoder, which improves performance but further increases memory consumption.
* T-SNE, for instance, is a non-parametric algorithm in this school and a parametric variant of t-SNE (van der Maaten, 2009) uses deep neural network to parametrize the embedding. The complexity  of t-SNE is O(n 2 ), where n is the number of data points, but it can be approximated in O(n log n) (van Der Maaten, 2014).
## 3 Experimentos
* KL divergence
* Cosine Similarity
* Similaridade por heroi, itens, gpm, xpm
* Quantidade de épocas, loss e validação e métricas de rede
### 3.1 Configuração
### 3.2 Análise
### 3.3 Resultados
## 4 Conclusões
