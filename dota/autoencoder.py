from dota.logger import get_logger
import keras as keras
from keras import layers, Model, optimizers, callbacks
from dota.logger import get_logger

log = get_logger("Dota2AE")


class Dota2AE(Model):
    def __init__(
        self,
        name: str,
        lr: float,
        dropout: float,
        early_stopping: bool,
        epochs: int,
        patience: int,
        batch_size: int,
        input_dim: int,
        latent_dim: int,
        encoder_layers: list[int],
        decoder_layers: list[int],
        embeddings_config: dict[str, tuple[int, int]],
    ):
        super(Dota2AE, self).__init__()

        # Nome do modelo
        self.name = f"Dota2AE_{name}"
        self.best_filename = f"Dota2AE_{name}_best.h5"
        self.loss_path = f"./Dota2AE_{name}_loss.csv"

        # Camadas de embedding para heróis, bans e estatísticas
        self.embeddings = embeddings_config
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.embeddings_config = embeddings_config

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learning_rate = lr
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        # Históricos de loss
        self.train_stopped = 0
        self.loss_val_history = []
        self.loss_history = []
        self.best_val_loss = float('inf')

    def _build_model(self):
        """Monta o modelo de autoencoder com embeddings, encoder e decoder."""
        log.separator()
        log.info("Construindo o modelo de autoencoder...")
        inputs = []
        concat_layers = []

        # Embeddings para features categóricas
        for name, (input_dim, output_dim) in self.embeddings_config.items():
            inp = layers.Input(shape=(1,), name=f"{name}_input")
            emb = layers.Embedding(input_dim=input_dim, output_dim=output_dim, name=f"{name}_emb")(inp)
            emb = layers.Flatten()(emb)
            inputs.append(inp)
            concat_layers.append(emb)

        # Input para outras features
        if self.input_dim > 0:
            cont_inp = layers.Input(shape=(self.input_dim,), name="cont_input")
            inputs.append(cont_inp)
            concat_layers.append(cont_inp)
        x = layers.Concatenate()(concat_layers) if len(concat_layers) > 1 else concat_layers[0]

        # Encoder
        for units in self.encoder_layers:
            x = layers.Dense(units, activation='relu')(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)

        # Decoder
        x = latent
        for units in self.decoder_layers:
            x = layers.Dense(units, activation='relu')(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        # Reconstrução
        output_dim = self.input_dim + sum([v[1] for v in self.embeddings_config.values()])
        output = layers.Dense(output_dim, activation='linear', name='reconstruction')(x)

        # Modelo
        self.model = Model(inputs=inputs, outputs=output, name=self.name)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        log.info(self.model.summary() or "Modelo criado com sucesso.")

    def fit_autoencoder(self, x, y=None, validation_data=None):
        """
        Treina o autoencoder. x deve ser um dicionário de entradas.
        """
        log.info("Iniciando treinamento do autoencoder...")
        cb = []
        if self.early_stopping:
            cb.append(callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True))
        history = self.model.fit(
            x, y if y is not None else x,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose="auto"
        )
        log.info("Treinamento finalizado.")
        return history

    def encode(self, x):
        """
        Retorna a representação latente para as entradas x.
        """
        encoder = Model(self.model.input, self.model.get_layer('latent').output)
        return encoder.predict(x)

    def decode(self, z):
        """
        Reconstrói a saída a partir do espaço latente z.
        """
        # Cria um modelo que recebe o latente e retorna a saída
        latent_input = layers.Input(shape=(self.latent_dim,))
        x = latent_input
        for units in self.decoder_layers:
            x = layers.Dense(units, activation='relu')(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        output_dim = self.input_dim + sum([v[1] for v in self.embeddings_config.values()])
        output = layers.Dense(output_dim, activation='linear')(x)
        decoder = Model(latent_input, output)
        return decoder.predict(z)

    def reconstruct(self, x):
        """
        Reconstrói a entrada x usando o autoencoder completo.
        """
        return self.model.predict(x)

    def train(self, train_data, val_data=None):
        """
        Treina o modelo com os dados de treino e validação.
        """
        log.info("Iniciando treinamento do modelo...")
        if val_data is not None:
            history = self.fit_autoencoder(train_data, validation_data=val_data)
        else:
            history = self.fit_autoencoder(train_data)
        
        self.loss_history.extend(history.history['loss'])
        if 'val_loss' in history.history:
            self.loss_val_history.extend(history.history['val_loss'])
        
        log.info("Treinamento concluído.")
        return history