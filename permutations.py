import itertools

learning_rate = [0.001]
dropout = [0.3]
hidden_layers = [
    [128, 32],
]
hero_pick_embedding_size = [8]
hero_role_embedding_size = [4]
latent_dimensions = [2, 4]

configs = list(itertools.product(
    learning_rate,
    dropout,
    hidden_layers,
    hero_pick_embedding_size,
    hero_role_embedding_size,
    latent_dimensions
))
from dota import Dota2
import permutations
from datetime import datetime
def run_permutations(_dota: Dota2, train_df, val_df, test_df, model_path, loss_path):
    print(f"Available permutations: {len(permutations.configs)}")
    # Recria o arquivo de resultados das permutações (limpa o conteúdo anterior)
    perm_best_loss = float('inf')
    perm_best_val_loss = float('inf')
    perm_best_mse = float('inf')
    best_perm_loss = None
    best_perm_val_loss = None
    results_filename = f"./tmp/permutations_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_filename, "w") as f:
        f.write("Dota2 Autoencoder Permutation Results\n")
        f.write("=" * 40 + "\n")
    for idx, perm in enumerate(permutations.configs):
        _epochs = 200
        print(f"Testing permutation {idx + 1}/{len(permutations.configs)}: {perm}")
        _dota.set_config({
            "learning_rate": perm[0],
            "dropout": perm[1],
            "hidden_layers": perm[2],
            "hero_pick_embedding_size": perm[3],
            "hero_role_embedding_size": perm[4],
            "latent_dimensions": perm[5],
        }, True)
        autoencoder, best_loss, best_val_loss =_dota.train_autoencoder(
            train_df, val_df, test_df, model_path, loss_path, epochs=_epochs, silent=True)
        autoencoder.save_loss_history(loss_path + f"_{idx + 1}.csv", True)
        autoencoder.save_model(model_path + f"_{idx + 1}.h5", True)
        accuracy, avg_mse, min_mse, max_mse = _dota.test_autoencoder(test_df, 0.1, silent=True)
        
        if best_loss < perm_best_loss:
            perm_best_loss = best_loss
            best_perm_loss = perm
        if best_val_loss < perm_best_val_loss:
            perm_best_val_loss = best_val_loss
            best_perm_val_loss = perm
        if avg_mse < perm_best_mse:
            perm_best_mse = avg_mse
        
        
        print(f"Permutation loss: {best_loss}, validation loss: {best_val_loss}")
        print(f"Accuracy: {accuracy}, Avg MSE: {avg_mse}, Min MSE: {min_mse}, Max MSE: {max_mse}")
        with open(results_filename, "a") as f:
            f.write(f"Permutation {idx + 1}/{len(permutations.configs)} Epochs {_epochs}: {perm}\n")
            f.write(f"Best loss: {best_loss}, Best validation loss: {best_val_loss}\n")
            f.write(f"Accuracy: {accuracy}, Avg MSE: {avg_mse}, Min MSE: {min_mse}, Max MSE: {max_mse}\n")
            f.write(f"Model saved to: {model_path}_{idx + 1}.h5\n")
            f.write("-" * 40 + "\n")
    print(f"Best permutation: {best_perm_loss} with loss {perm_best_loss}")
    print(f"Best validation permutation: {best_perm_val_loss} with loss {perm_best_val_loss}")
    print(f"Best MSE: {perm_best_mse}")