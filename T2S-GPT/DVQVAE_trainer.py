# T2s-GPT/DVQVAE_trainer.py
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.DVQVAE import DVQVAE_Encoder, DVQVAE_Decoder, DVQVAELoss
from dataset.random_dataset import RandomDataset
from dataset.phoenix_dataset import SignLanguageDataset
from utils.file_utils import get_unique_path
from utils.pad_seq import pad_collate_fn

def train_dvqvae_model(num_epochs=10, batch_size=32, sign_language_dim=512,
                       T=100, latent_dim=512, vocab_size=500, codebook_size=1024, 
                       output_dim=512):
    # Prepare dataset and data loader

    ## RandomDataset
    dataset = RandomDataset(T, sign_language_dim, output_dim, vocab_size, num_samples=5)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ## SignLanguageDataset
    dataset = SignLanguageDataset()

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # Initialize models, loss function, optimizer, and scheduler
    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size)
    decoder = DVQVAE_Decoder(latent_dim, output_dim)
    loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    train_loss_list = []
    val_loss_list = []
    loss_path = get_unique_path('./data/DVQVAE_loss.txt')
    model_path = get_unique_path('./trained_model/dvqvae_model.pth')

    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        total_train_loss = 0.0
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence

            # Forward pass through DVQ-VAE
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=True)
            X_re = decoder(Z_quantized, D_T_l, H_T)

            # For demonstration, replace this with actual values
            P_Y_given_X_re = torch.ones(batch_size, output_dim, T).to(device)

            train_loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re, loss_path)

            # Back Propagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_train_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        train_loss_list.append(total_train_loss / len(train_loader))


    # Save the model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)

    return train_loss_list, X_T, X_re, loss_path

def evaluate_model(encoder, decoder, val_loader, output_dim=512):
    encoder.eval()
    decoder.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_T = batch['sign_language_sequence'].to(device)  # Input
            Y = batch['spoken_language_text'].to(device)       # Ground truth

            # Forward pass
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=False)
            X_re = decoder(Z_quantized, D_T_l, H_T)

            # Probabilistic output for predicted sequence
            P_Y_given_X_re = torch.ones(X_T.size(0), output_dim, T).to(device)

            # Calculate validation loss
            loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12)
            loss_path = get_unique_path('./data/DVQVAE_loss.txt')
            val_loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re, loss_path)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss after Training: {avg_val_loss:.4f}")
    
    return avg_val_loss


def plot_loss(loss_file):
    L_X_re_list = []
    L_vq_list = []
    L_budget_list = []
    # L_slt_list = []
    L_total_list = []
    
    with open(loss_file, 'r') as f:
        for line in f:
            # Split the line into values
            values = line.strip().split(',')
            L_X_re, L_vq, L_budget, L_total = map(float, values)
            # L_X_re, L_vq, L_budget, L_slt, L_total = map(float, values)
            L_X_re_list.append(L_X_re)
            L_vq_list.append(L_vq)
            L_budget_list.append(L_budget)
            # L_slt_list.append(L_slt)
            L_total_list.append(L_total)
    
    iterations = list(range(1, len(L_X_re_list) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.ylim(0,10)
    plt.plot(iterations, L_X_re_list, label='L_X_re')
    plt.plot(iterations, L_vq_list, label='L_vq')
    plt.plot(iterations, L_budget_list, label='L_budget')
    # plt.plot(iterations, L_slt_list, label='L_slt')
    plt.plot(iterations, L_total_list, label='L_total')
    
    plt.title('DVQ-VAE Training Loss Components')
    plt.grid(True)
    plt.legend(loc='best')
    
    plot_dir = './data'
    plot_file = os.path.join(plot_dir, 'DVQVAE_plot.png')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    


def main():
    loss_list, X_T, X_re, loss_path = train_dvqvae_model(num_epochs=20, batch_size=5, codebook_size=64, sign_language_dim=150)
    plot_loss(loss_path)

if __name__ == "__main__":
    main()
