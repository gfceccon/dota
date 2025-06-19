import keras as keras
from keras import layers, Model, optimizers, callbacks
from dota.logger import get_logger

log = get_logger("Dota2AE")


class Dota2AEKeras(Model):
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
        """
        Autoencoder Keras para Dota2.
        embeddings_config: dict com nome da feature -> (input_dim, output_dim)
        """
        super().__init__()
        self.name = name
        self.lr = lr
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.embeddings_config = embeddings_config
        self._build_model()

    def _build_model(self):
        """Monta o modelo de autoencoder com embeddings, encoder e decoder."""
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

        # Input para features contínuas
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
            optimizer=optimizers.Adam(learning_rate=self.lr),
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
