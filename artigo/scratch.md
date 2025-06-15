# Semelhança entre Partidas de Dota 2 usando Autoencoders e Clustering


## Resumo
* Resumo. Este trabalho propõe a aplicação de autoencoder, juntamente com um classificador, para detectar semelhança entre partidas profissionais de Dota 2, um jogo eletrônico do gênero MOBA. A análise baseia-se em características como a composição dos heróis, desempenho econômico, tempo de partida e objetivos alcançados. O estudo é focado em analisar modelos, treinar em diferentes versões de jogo e comparar partidas de campeonatos. O modelo usa técnicas para redução de dimensões em heróis escolhidos e banidos, entre outras variáveis categóricas.


## 1 Introdução
* Assuntos
    * Dinheiro
    * Dados ricos,
    * Aplicação matemática e estatística
    * Estudo social e comportamental
    * Sponsors de análise de partida
    * Envolvimento da OpenAPI https://openai.com/index/dota-2/
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
* Open Dota https://docs.opendota.com
* Kaggle https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023/data
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
* Autoencoder com clustering vantagens e desvantagens
* Dificuldade de analisar dados após codificação do espaço latente
* Pode não encontrar cluster não relevantes
* Fine-tunning
* Tipo de Autoencoders
    * Autoencoders padrão
    * VAE - Variational Autoencoders
    * Sparse Autoencoders
    * SAE - Stacked Autoencoders
    * Deep Embedded Clustering (DEC)
* DEC foi usado por sua capacidade de agrupar melhor os clusters
* Cálculo de diferença usando Kullback–Leibler (KL) divergence
### 2.3 Clustering
* DEC precisa de número de cluster definido
    * Semi-supervisionado para pré treinamento dos pesos
        * Label de jogos, trabalho manual
    * Usado MINIST etc. onde já existe o número
* k-means e Gaussian Mixture Models (GMM)
    * Quantidade de cluster não definida
* HDBSCAN e OPTICS
    * Cálculo do Epsilon baseado na distâncias de vizinhos
* Spectral Clustering
    * Não linear, espaço não euclidiano
    * Dificultar visualização de dados
    * Já foi usado spectral clustering e embedding, problemas de memória.
* T-SNE
    * Deep neural network to parametrize the embedding.
## 3 Experimentos
* TODO
### 3.1 Configuração
* Analise de jogos de The International para maior relevância (meta do jogo)
* KL divergence para treinamento de rede
* Cosine Similarity para similaridade entre jogos
* Similaridade por heróis, itens, gpm, xpm, observer, sentry
* Quantidade de épocas, loss e validação e métricas de rede
### 3.2 Análise
* Analise de cluster usando k-means e OPTICIS
* Analise de 
* TODO
### 3.3 Resultados
* TODO
## 4 Conclusões
* TODO
