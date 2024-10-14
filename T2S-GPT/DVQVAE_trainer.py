import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.DVQVAE import DVQVAE_Encoder, DVQVAE_Decoder, DVQVAELoss
from dataset.random_dataset import RandomDataset

def train_dvqvae_model(num_epochs=10, batch_size=32, sign_language_dim=512,
                       T=100, latent_dim=512, vocab_size=500, codebook_size=1024, 
                       output_dim=512):
    # Prepare dataset and data loader
    dataset = RandomDataset(T, sign_language_dim, output_dim, vocab_size, num_samples=30)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models, loss function, optimizer, and scheduler
    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size)
    decoder = DVQVAE_Decoder(latent_dim, output_dim)
    loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    loss_list = []

    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        total_loss = 0.0
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence

            # Forward pass through DVQ-VAE
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T)
            X_re = decoder(Z_quantized, D_T_l, H_T)

            # For demonstration, replace this with actual values
            P_Y_given_X_re = torch.rand(batch_size, output_dim, T).to(device)

            loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re)

            # Back Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        loss_list.append(total_loss / len(train_loader))

    # Save the model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'dvqvae_model.pth')

    return loss_list

def plot_loss(loss_list):
    # Plot loss
    plot_dir = '../visualization'
    plot_file = os.path.join(plot_dir, 'DVQVAE_plot.png')
    os.makedirs(plot_dir, exist_ok=True)
    indices = list(range(len(loss_list)))
    plt.plot(indices, loss_list, linestyle='--', color='b', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('DVQ-VAE Training Loss')
    plt.grid()
    plt.savefig(plot_file)
    plt.close()

def main():
    loss_list = train_dvqvae_model(num_epochs=10, batch_size=32)
    plot_loss(loss_list)

if __name__ == "__main__":
    main()
