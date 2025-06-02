# Exemplos de Uso do Dota Autoencoder

Este diretório contém exemplos práticos de como usar o autoencoder para análise de partidas de Dota 2.

## Arquivos de Exemplo

### 1. `example_autoencoder_usage.py` - Exemplo Completo
Um exemplo abrangente que demonstra todas as funcionalidades do autoencoder:

- ✅ **Configuração e carregamento de dados**
- ✅ **Treinamento completo com validação**
- ✅ **Visualização do progresso de treinamento**
- ✅ **Análise do espaço latente com PCA**
- ✅ **Busca por similaridade entre partidas**
- ✅ **Detecção de anomalias**
- ✅ **Análise de qualidade de reconstrução**
- ✅ **Salvamento e carregamento de modelos**

**Como executar:**
```bash
python example_autoencoder_usage.py
```

**Saídas geradas:**
- `training_history.png` - Gráfico do histórico de treinamento
- `latent_space_pca.png` - Visualização 2D do espaço latente
- `dota_autoencoder_model.pth` - Modelo treinado salvo

### 2. `quick_autoencoder_examples.py` - Exemplos Rápidos
Exemplos focados e diretos para casos de uso específicos:

- 🚀 **Treinamento rápido** - Como treinar rapidamente o modelo
- 🔍 **Análise de similaridade** - Encontrar partidas similares
- 🚨 **Detecção de anomalias** - Identificar partidas incomuns
- 🎯 **Extração de features** - Obter representações latentes
- 🎲 **Clustering** - Agrupar partidas similares
- 💾 **Persistência** - Salvar/carregar modelos

**Como executar:**
```bash
python quick_autoencoder_examples.py
```

## Casos de Uso Demonstrados

### 1. 🧠 Treinamento do Autoencoder
```python
from autoencoders import create_autoencoder_from_dataset

# Criar modelo automaticamente do dataset
model, processor = create_autoencoder_from_dataset(dataset_path)

# Treinar
trainer = DotaAutoencoderTrainer(model)
history = train_autoencoder(model, trainer, train_loader, val_loader)
```

### 2. 🔍 Análise de Similaridade
```python
# Extrair representações latentes
latent_representations = extract_latent_representations(model, data_loader)

# Encontrar partidas similares
similar_indices, similarities = find_similar_matches(
    latent_representations, 
    match_index=0, 
    top_k=5
)
```

### 3. 🚨 Detecção de Anomalias
```python
# Detectar partidas anômalas
anomaly_indices = detect_anomalies(
    latent_representations, 
    threshold_percentile=95
)
```

### 4. 📊 Visualização do Espaço Latente
```python
# Visualizar com PCA
analyze_latent_space(latent_representations, save_path="pca_plot.png")
```

### 5. 💾 Persistência de Modelos
```python
# Salvar modelo completo
save_model(model, processor, "my_model.pth")

# Carregar modelo
loaded_model, loaded_processor = load_model("my_model.pth")
```

## Configuração e Requisitos

### Dependências
As dependências estão definidas no `pyproject.toml`:
- `torch` - PyTorch para deep learning
- `numpy` - Computação numérica
- `polars` - Manipulação de dados
- `scikit-learn` - Preprocessing e métricas
- `matplotlib` - Visualização
- `kagglehub` - Download do dataset

### Instalação
```bash
# Se usando uv (recomendado)
uv sync

# Ou com pip
pip install -e .
```

## Estrutura dos Dados

O autoencoder processa dados de partidas de Dota 2 com:

### Entrada
- **Heróis Radiant**: 5 heróis do time Radiant
- **Heróis Dire**: 5 heróis do time Dire  
- **Heróis Banidos**: Até 14 heróis banidos na fase de draft
- **Estatísticas**: Métricas de performance dos jogadores
  - Kills, Deaths, Assists
  - Gold per minute, XP per minute

### Saída
- **Representação Latente**: Vetor compacto representando a partida
- **Reconstrução**: Reconstrução das features originais

## Parâmetros Importantes

### Modelo
```python
DotaMatchAutoencoder(
    max_heroes=150,           # Número máximo de heróis
    embedding_dim=32,         # Dimensão dos embeddings
    hidden_dims=[256,128,64], # Camadas ocultas
    latent_dim=32,           # Dimensão do espaço latente
    dropout_rate=0.2         # Taxa de dropout
)
```

### Treinamento
```python
DotaAutoencoderTrainer(
    model=model,
    learning_rate=1e-3,      # Taxa de aprendizado
    weight_decay=1e-5        # Regularização L2
)
```

## Dicas de Uso

### 🎯 Para Melhores Resultados
1. **Use mais dados**: Quanto mais partidas, melhor a qualidade das representações
2. **Ajuste hiperparâmetros**: Experimente diferentes dimensões latentes e arquiteturas
3. **Normalize adequadamente**: O preprocessamento é crucial para convergência
4. **Monitore overfitting**: Use validação para evitar sobreajuste

### 🔧 Customização
- **Modifique `hidden_dims`** para arquiteturas diferentes
- **Ajuste `latent_dim`** baseado na complexidade desejada
- **Customize `dropout_rate`** para controlar regularização
- **Implemente novas métricas** de loss se necessário

### 📊 Análise Downstream
As representações latentes podem ser usadas para:
- **Clustering** de estilos de jogo
- **Classificação** de resultados
- **Recomendação** de estratégias
- **Detecção** de padrões meta

## Solução de Problemas

### Erro de Memória
```python
# Reduza o batch size
train_loader = DataLoader(dataset, batch_size=16)  # ao invés de 32

# Ou use menos dados
sample_data = dataset.sample(500)  # ao invés do dataset completo
```

### Convergência Lenta
```python
# Aumente a learning rate
trainer = DotaAutoencoderTrainer(model, learning_rate=1e-2)

# Ou ajuste a arquitetura
model = DotaMatchAutoencoder(hidden_dims=[128, 64])  # Rede menor
```

### Resultados Ruins
1. **Verifique os dados**: Certifique-se que não há valores faltantes
2. **Normalize features**: Especialmente as estatísticas numéricas
3. **Aumente épocas**: Modelo pode precisar de mais treinamento
4. **Experimente hiperparâmetros**: Diferentes configurações podem funcionar melhor

## Contribuindo

Para adicionar novos exemplos:
1. Crie um novo arquivo `.py` com exemplos específicos
2. Documente claramente o caso de uso
3. Inclua comentários explicativos
4. Teste com dados reais
5. Atualize este README

---

**Nota**: Os exemplos usam uma amostra pequena dos dados para execução rápida. Para resultados em produção, use o dataset completo e treine por mais épocas.
