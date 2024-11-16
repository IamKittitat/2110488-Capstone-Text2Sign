# T2s-GPT/trainer.py
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.DVQVAE import DVQVAE_Encoder, DVQVAE_Decoder, DVQVAELoss
from models.T2S_GPT import T2S_GPT, T2SGPTLoss
from dataset.random_dataset import RandomDataset
from dataset.phoenix_dataset import SignLanguageDataset
from utils.file_utils import get_unique_path
from utils.visualization import plot_loss
from utils.pad_seq import pad_collate_fn

def train_both_model(dvq_epochs=10, t2s_epoch = 10, batch_size=32, sign_language_dim=150,
                       T=100, latent_dim=512, vocab_size=500, codebook_size=1024, 
                       output_dim=512):
    ## RandomDataset
    dataset = RandomDataset(T, sign_language_dim, output_dim, vocab_size, num_samples=5)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ## SignLanguageDataset
    # dataset = SignLanguageDataset()
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size)
    decoder = DVQVAE_Decoder(latent_dim, output_dim)
    loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=dvq_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    loss_list = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_dir = get_unique_path(os.path.join(current_dir, 'result/trainer'))
    os.makedirs(folder_dir, exist_ok=True)
    dvq_loss_path = os.path.join(folder_dir, 'DVQVAE_loss.txt')
    dvq_model_path = os.path.join(folder_dir, 'DVQVAE_model.pth')

    # Training loop
    for epoch in range(dvq_epochs):
        encoder.train()
        decoder.train()

        total_loss = 0.0
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence

            # Forward pass through DVQ-VAE
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=True)
            X_re = decoder(Z_quantized, D_T_l, H_T)

            # For demonstration, replace this with actual values
            P_Y_given_X_re = torch.ones(batch_size, output_dim, T).to(device)

            loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re, dvq_loss_path)

            # Back Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{dvq_epochs}], Loss: {total_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        loss_list.append(total_loss / len(train_loader))

    # Save the model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, dvq_model_path)

    ##############################################################################################################################
    # T2S-GPT 
    model = T2S_GPT(vocab_size=vocab_size, embed_dim=64, codebook_size=codebook_size, max_duration=100, device=device).to(device)
    t2sgpt_loss = T2SGPTLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = torch.load(dvq_model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    t2s_loss_path = os.path.join(folder_dir, 'T2SGPT_loss.txt')
    t2s_model_path = os.path.join(folder_dir, 'T2SGPT_model.pth')

    for epoch in range(t2s_epoch):
        model.train()
        total_loss, total_code_loss, total_duration_loss = 0, 0, 0
        
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence
            
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=False)
            S_T_pred, S_T_expected, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits = model(Y, codebook_indices, D_T_l)
            loss_total = t2sgpt_loss(code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected, t2s_loss_path)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
        
        print(f"Epoch {epoch + 1}/{t2s_epoch}, Total Loss: {total_loss:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, t2s_model_path)

    return folder_dir


def main():
    folder_dir = train_both_model(dvq_epochs=10, t2s_epoch=3, batch_size=5, codebook_size=64)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dvq_loss_path = os.path.join(folder_dir, 'DVQVAE_loss.txt')
    dvq_save_path = os.path.join(folder_dir, 'DVQVAE_plot.png')
    t2s_loss_path = os.path.join(folder_dir, 'T2SGPT_loss.txt')
    t2s_save_path = os.path.join(folder_dir, 'T2SGPT_plot.png')
    plot_loss(dvq_loss_path, ["L_X_re", "L_vq", "L_budget", "L_total"], "DVQ-VAE Training Loss", dvq_save_path)
    plot_loss(t2s_loss_path, ["L_code", "L_duration", "L_total"], "T2S-GPT Training Loss", t2s_save_path, y_lim = 10000)

if __name__ == "__main__":
    main()
