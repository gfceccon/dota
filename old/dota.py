import ast
import os
import kagglehub
import torch
from heroes import get_heroes
from dataset import get_dataset
from patches import get_patches
from model import Dota2Autoencoder
import polars as pl
from plot import plot_loss_history
from datetime import datetime
import random
import numpy as np
import torch


class Dota2:
    def __init__(self, patches: list[int], tier: list[str] = ["professional"],
                 duration: tuple[int, int] = (30, 120), silent: bool = False, name: str = "dota2"):
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        pl.set_random_seed(self.seed)
        self.silent = silent
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.name = name
        self.base_filename = f"{self.name}_{timestamp}"
        self.log_filename = f"log_{self.base_filename}.txt"
        self._log("="*50)
        self._log("Inicializando Dota2 Autoencoder...")
        self.n_players = 5
        self.n_bans = 7
        self.patches = patches
        self.tier = tier
        self.duration = duration
        self.mse_threshold = 0.1

        path = kagglehub.dataset_download(
            "bwandowando/dota-2-pro-league-matches-2023")

        os.makedirs("tmp", exist_ok=True)
        os.makedirs("best", exist_ok=True)

        self.path = path
        self.should_train = True
        self.configuration = {
            "hero_pick_embedding_dim": 8,
            "hero_role_embedding_dim": 4,
            "latent_dim": 2,
            "hidden_layers": [128, 64, 32],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "mse_threshold": 0.1,
            "verbose": False
        }

        self.dataset, self.player_cols, self.hero_cols = get_dataset(
            path, tier, duration, patches)

        self.get_filenames(patches)
        self.heroes, _, _, self.dict_roles = get_heroes(path)
        self.n_heroes = self.heroes.select("hero_idx").count().collect().item()
        self.n_hero_stats = len(self.dict_roles) + len(self.hero_cols)
        self.n_player_stats = len(self.player_cols)
        self.patches_info = get_patches(path)

        self.autoencoder = self.create_autoencoder()

    def _log(self, *args, **kwargs):
        log_message = ' '.join(str(arg) for arg in args)
        with open(f"./tmp/{self.log_filename}", "a", encoding="utf-8") as log_file:
            log_file.write(log_message + "\n")
        if not getattr(self, "silent", False):
            print(*args, **kwargs)

    def create_autoencoder(self):
        self.autoencoder = Dota2Autoencoder(
            hero_pick_embedding_dim=self.configuration["hero_pick_embedding_dim"],
            hero_role_embedding_dim=self.configuration["hero_role_embedding_dim"],
            dict_roles=self.dict_roles,
            hero_cols=self.hero_cols,
            player_cols=self.player_cols,
            n_heroes=self.n_heroes + 1,
            n_players=self.n_players,
            n_bans=self.n_bans,
            latent_dim=self.configuration["latent_dim"],
            hidden_layers=self.configuration["hidden_layers"],
            dropout=self.configuration["dropout"],
            learning_rate=self.configuration["learning_rate"],
            verbose=self.configuration["verbose"],
            log_filename=self.log_filename,
        )
        return self.autoencoder

    def set_config(self, config: dict, silent: bool = False):
        self.silent = silent
        self._log("="*50)
        self._log("Configurando Dota2 Autoencoder com novos parâmetros")
        # Atualiza a configuração com os valores fornecidos em config, mantendo os valores atuais caso não estejam presentes
        keys = [
            "hero_pick_embedding_dim",
            "hero_role_embedding_dim",
            "latent_dim",
            "hidden_layers",
            "dropout",
            "learning_rate",
            "mse_threshold",
            "verbose"
        ]
        for key in keys:
            if key in config:
                self.configuration[key] = config[key]

        # Atualiza o threshold de MSE separadamente, se fornecido
        if "mse_threshold" in config:
            self.mse_threshold = config["mse_threshold"]

        # Marca que o modelo deve ser treinado novamente após alteração de configuração
        self.should_train = True
        self.create_autoencoder()

    def train_autoencoder(self, epochs=100, silent: bool = False):
        self.silent = silent
        train_data, validation_data, test_data = self.prepare_data_splits(
            self.dataset, silent=self.silent)

        self._log("="*50)
        self._log("Iniciando treinamento do Dota2 Autoencoder...")
        self._log(
            f"Tamanho dos dados de treino: {train_data.shape[0]}, Tamanho dos dados de validação: {validation_data.shape[0]}, Tamanho dos dados de teste: {test_data.shape[0]}")
        self._log(
            f"Total de estatísticas de heróis: {self.n_hero_stats}, Total de estatísticas de jogadores: {self.n_player_stats}")

        self._log("="*50)
        self._log("Treinando Dota2 Autoencoder...")
        self.autoencoder.train_data(train_data, validation_data, epochs=epochs,
                                    verbose=self.configuration["verbose"], silent=self.silent)
        self.autoencoder.save_loss_history(
            self.loss_history_path, silent=self.silent)
        self.autoencoder.save_model(self.model_path, silent=self.silent)

        self._log("="*50)
        self._log(
            f"Modelo salvo em {self.model_path} e histórico de perda salvo em {self.loss_history_path}.")
        self._log("Treinamento concluído.")
        return self.autoencoder, self.autoencoder.best_val_loss

    def test_autoencoder(self, test_data: pl.DataFrame, mse_threshold: float = 0.1, silent: bool = False):
        self.silent = silent
        self._log("="*50)
        self._log("Iniciando teste do Dota2 Autoencoder...")
        self._log(f"Tamanho dos dados de teste: {test_data.shape[0]}")
        self._log(f"Threshold de MSE: {mse_threshold}")

        accuracy, avg_mse, min_mse, max_mse = self.autoencoder.test_model(
            test_data, threshold=mse_threshold)

        self._log("="*50)
        self._log(f"Acurácia: {accuracy:.2f} (Threshold {mse_threshold})")
        self._log(f"MSE médio: {avg_mse:.6f}")
        self._log(f"MSE mínimo: {min_mse:.6f}")
        self._log(f"MSE máximo: {max_mse:.6f}")
        return accuracy, avg_mse, min_mse, max_mse

    def load_or_prepare_dataset(self, dataset_path: str = "", metadata_path: str = "", silent: bool = False):
        self.silent = silent
        self._log("="*50)
        self._log("Carregando ou preparando o dataset...")
        if (os.path.exists(dataset_path) and os.path.exists(metadata_path)):
            dataset = pl.read_json(dataset_path)
            dataset_metadata = pl.read_json(metadata_path)
            self.n_heroes = dataset_metadata["n_heroes"].item()
            self.n_hero_stats = dataset_metadata["n_hero_stats"].item()
            self.n_player_stats = dataset_metadata["n_player_stats"].item()
            self.player_cols = dataset_metadata["player_cols"].item()
            self.hero_cols = dataset_metadata["hero_cols"].item()
            self.dict_roles = dataset_metadata["self.dict_roles"].item()
            self.patches_info = ast.literal_eval(
                dataset_metadata["patches_info"].item())
            self._log("Dataset já carregado do arquivo.")
            return dataset, False
        else:
            dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
            path = kagglehub.dataset_download(dataset_name)
            dataset, self.player_cols, self.hero_cols = get_dataset(
                path, specific_patches=self.patches)
            self.heroes, _, _, self.dict_roles = get_heroes(path)
            self.n_heroes = self.heroes.select(
                "hero_idx").count().collect().item()
            self.n_hero_stats = len(self.dict_roles) + len(self.hero_cols)
            self.n_player_stats = len(self.player_cols)
            self.patches_info = get_patches(path)
            self._log("Dataset carregado e pré-processado com sucesso!")
            return dataset, True

    def save_dataset_and_metadata(self, dataset: pl.DataFrame, dataset_path: str, metadata_path: str, silent: bool = False):
        self.silent = silent
        self._log("="*50)
        print(f"Salvando dataset em {dataset_path}...")
        dataset.write_json(dataset_path)
        dataset_metadata = pl.DataFrame({
            "n_heroes": [self.n_heroes],
            "n_hero_stats": [self.n_hero_stats],
            "n_player_stats": [self.n_player_stats],
            "player_cols": [self.player_cols],
            "hero_cols": [self.hero_cols],
            "self.dict_roles": [self.dict_roles],
            "patches_info": [str(self.patches_info)]
        })
        dataset_metadata.write_json(metadata_path)
        self._log(f"Dataset completo salvo em {dataset_path}.")
        self._log(f"Metadados do dataset salvos em {metadata_path}.")

    def prepare_data_splits(self, dataset: pl.DataFrame, df_scale=1.0, silent: bool = False):
        self.silent = silent
        train_data = dataset.sample(
            fraction=0.75 * df_scale, shuffle=True)
        validation_data = dataset.sample(
            fraction=0.15 * df_scale, shuffle=True)
        test_data = dataset.sample(
            fraction=0.15 * df_scale, shuffle=True)
        return train_data, validation_data, test_data

    def get_filenames(self, patches: list[int], silent: bool = False) -> dict[str, str]:
        self.silent = silent
        patches_str = [str(patch) for patch in patches]
        filename = f"{self.base_filename}_{'_'.join(patches_str)}"

        self.filename = filename
        self.dataset_path = f"./tmp/{filename}_dataset.json"
        self.metadata_path = f"./tmp/{filename}_metadata.json"
        self.model_path = f"./tmp/{filename}_model.h5"
        self.loss_history_path = f"./tmp/{filename}_history.csv"
        self.plot_path = f"./tmp/{filename}_history.png"
        self.report_path = f"./tmp/{filename}_report.txt"

        return {
            "save_filename": self.filename,
            "dataset_full_path": self.dataset_path,
            "dataset_metadata_path": self.metadata_path,
            "save_model_path": self.model_path,
            "save_loss_history_path": self.loss_history_path,
            "save_plot_loss_history_path": self.plot_path,
            "report_path": self.report_path,
        }

    def save_report(self, train_data: pl.DataFrame, validation_data: pl.DataFrame, test_data: pl.DataFrame,
                    report_path: str, loss_history_path: str, plot_path: str, silent: bool = False):
        self.silent = silent
        self._log("="*50)
        self._log("Salvando relatório de desempenho do modelo...")

        used_patches = [patch_name for patch_id,
                        (_, patch_name) in self.patches_info.items() if patch_id in self.patches]

        lines = [
            "="*60,
            "RELATÓRIO DE DESEMPENHO DO MODELO DOTA 2",
            "="*60,
            "",
            "[DADOS DO CONJUNTO]",
            f"- Treinamento: {train_data.shape[0]} amostras",
            f"- Validação: {validation_data.shape[0]} amostras",
            f"- Teste: {test_data.shape[0]} amostras",
            f"- Threshold de MSE: {self.mse_threshold}",
            f"- Patches: {', '.join(str(patch) for patch in used_patches)}",
            "",
            "[ARQUIVOS GERADOS]",
            f"- Modelo salvo: {self.model_path}",
            f"- Histórico de perda: {self.loss_history_path}",
            f"- Plot de perda: {self.plot_path}",
            "",
            "[ESTRUTURA DOS DADOS]",
            f"- Colunas de heróis: {', '.join(self.autoencoder.hero_columns)}",
            f"- Tamanho das colunas de heróis: {len(self.autoencoder.hero_columns)}",
            f"- Colunas de jogadores: {', '.join(self.autoencoder.player_columns)}",
            f"- Tamanho das colunas de jogadores: {len(self.autoencoder.player_columns)}",
            "",
            "[ARQUITETURA DO MODELO]",
            f"- Dimensão de embedding (picks de heróis): {self.autoencoder.hero_pick_embedding_dim}",
            f"- Dimensão de embedding (roles de heróis): {self.autoencoder.hero_role_embedding_dim}",
            f"- Dimensões de entrada: {self.autoencoder.input_dim}",
            f"- Dimensão latente: {self.autoencoder.latent_dim}",
            f"- Arquitetura: {([self.autoencoder.input_dim] + self.autoencoder.hidden_layers +
                               [self.autoencoder.latent_dim] +
                               list(reversed(self.autoencoder.hidden_layers)) + [self.autoencoder.input_dim])}",
        ]
        lines.append("")
        lines.append("[INFORMAÇÕES DO TREINAMENTO]")
        lines.append(f"- Épocas: {self.autoencoder.train_stopped}")
        lines.append(
            f"- Perda de treino final: {self.autoencoder.loss_history[-1]:.6f}")
        lines.append(
            f"- Perda de validação final: {self.autoencoder.val_loss_history[-1]:.6f}")
        lines.append("")
        accuracy, avg_mse, min_mse, max_mse = self.autoencoder.test_model(
            test_data, threshold=self.mse_threshold)
        lines.append("[RESULTADOS NO CONJUNTO DE TESTE]")
        lines.append(
            f"- Acurácia: {accuracy:.2f} (Threshold {self.mse_threshold})")
        lines.append(f"- MSE médio: {avg_mse:.6f}")
        lines.append(f"- MSE mínimo: {min_mse:.6f}")
        lines.append(f"- MSE máximo: {max_mse:.6f}")
        lines.append("")
        lines.append("[INFORMAÇÕES DOS PATCHES]")
        for patch_id, (patch_count, patch_name) in self.patches_info.items():
            lines.append(f"- Patch {patch_id} ({patch_name}): {patch_count}")
        lines.append("")
        lines.append("[ROLES DISPONÍVEIS]")
        for role_name, role_id in self.autoencoder.dict_roles.items():
            lines.append(f"- {role_name}: {role_id}")

        lines.append("")
        lines.append("Teste do modelo concluído.")
        lines.append("="*60)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\r\n".join(lines))

        self._log(f"Relatório salvo em {report_path}.")
        self._log(
            f"Plotando histórico de perda em {self.plot_path}...")
        plot_loss_history(loss_history_path, plot_path,
                          title="Dota 2 Autoencoder Loss History")

    def print_architecture(self, silent: bool = False):
        self.silent = silent
        input_dim = self.autoencoder.input_dim
        self._log("="*50)
        self._log("Arquitetura do Dota2 Autoencoder:")
        self._log(
            f"Dimensão de embedding (picks de heróis): {self.configuration['hero_pick_embedding_dim']}")
        self._log(
            f"Dimensão de embedding (roles de heróis): {self.configuration['hero_role_embedding_dim']}")
        self._log(f"Dimensão latente: {self.configuration['latent_dim']}")
        self._log(
            f"Arquitetura: {(
                [input_dim] + self.configuration['hidden_layers'] +
                [self.configuration['latent_dim']] +
                list(reversed(self.configuration['hidden_layers'])) + [input_dim])
            })")

    def load_patch(self, patches: list[int], silent: bool = False):
        self.silent = silent
        self._log("="*50)
        self._log("Carregando patches...")
        self.patches = patches
        self.get_filenames(patches)
        self.patches_info = get_patches(self.path)
        self._log(f"Patches carregados: {self.patches_info}")
        self.dataset, self.player_cols, self.hero_cols = get_dataset(
            self.path, specific_patches=self.patches)
        self.heroes, _, _, self.dict_roles = get_heroes(self.path)
        self.n_heroes = self.heroes.select("hero_idx").count().collect().item()
        self.n_hero_stats = len(self.dict_roles) + len(self.hero_cols)
        self.n_player_stats = len(self.player_cols)
        self.autoencoder = self.create_autoencoder()
        return self.patches_info
    
if __name__ == "__main__":
    dota2 = Dota2(patches=[56])
    print(dota2.autoencoder.input_dim)