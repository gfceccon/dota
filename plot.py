import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(csv_file_path, save_path=None, title="Training and Validation Loss"):
    df = pl.read_csv(csv_file_path)
    
    epochs = np.arange(1, len(df) + 1)
    train_loss = df['loss'].to_numpy()
    val_loss = df['eval_loss'].to_numpy()
    
    min_train_loss = float(train_loss.min())
    min_val_loss = float(val_loss.min())
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_loss, 'b-', label=f'Training Loss (min: {min_train_loss:.4f})', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_loss, 'r-', label=f'Validation Loss (min: {min_val_loss:.4f})', linewidth=2, marker='s', markersize=4)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
if __name__ == "__main__":
    csv_path = "tmp/dota2_autoencoder_loss_history.csv"
    save_path = "tmp/dota2_autoencoder_loss_plot.png"
    
    plot_loss_history(csv_path, save_path, title="Dota 2 Autoencoder Loss History")