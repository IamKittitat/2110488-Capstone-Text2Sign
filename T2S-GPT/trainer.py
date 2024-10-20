import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.DVQVAE import DVQVAE_Encoder, DVQVAE_Decoder, DVQVAELoss
from models.T2S_GPT import T2S_GPT, T2SGPTLoss
from dataset.random_dataset import RandomDataset
from utils.file_utils import get_unique_path

def train_both_model(num_epochs=10, batch_size=32, sign_language_dim=512,
                       T=100, latent_dim=512, vocab_size=500, codebook_size=1024, 
                       output_dim=512):
    dataset = RandomDataset(T, sign_language_dim, output_dim, vocab_size, num_samples=5)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size)
    decoder = DVQVAE_Decoder(latent_dim, output_dim)
    loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    loss_list = []
    loss_path = get_unique_path('./data/DVQVAE_loss.txt')
    model_path = get_unique_path('./trained_model/dvqvae_model.pth')

    # Training loop
    for epoch in range(num_epochs):
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

            loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re, loss_path)

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
    }, model_path)

    ##############################################################################################################################
    # T2S-GPT 
    model = T2S_GPT(vocab_size=vocab_size, embed_dim=64, codebook_size=codebook_size, max_duration=100, device=device).to(device)
    t2sgpt_loss = T2SGPTLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    t2sgpt_loss_path = get_unique_path('./data/T2SGPT_loss.txt')
    t2sgpt_model_path = get_unique_path('./trained_model/t2sgpt_model.pth')


    for epoch in range(20):
        model.train()
        total_loss, total_code_loss, total_duration_loss = 0, 0, 0
        
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence
            
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=False)
            S_T_pred, S_T_expected, H_code, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits = model(Y, codebook_indices, D_T_l)
            loss_total = t2sgpt_loss(code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected, t2sgpt_loss_path)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
        
        print(f"Epoch {epoch + 1}/{20}, Total Loss: {total_loss:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, t2sgpt_model_path)

    return loss_path, t2sgpt_loss_path


def plot_loss(loss_file):
    L_X_re_list = []
    L_vq_list = []
    L_budget_list = []
    L_slt_list = []
    L_total_list = []
    
    with open(loss_file, 'r') as f:
        for line in f:
            # Split the line into values
            values = line.strip().split(',')
            L_X_re, L_vq, L_budget, L_slt, L_total = map(float, values)
            L_X_re_list.append(L_X_re)
            L_vq_list.append(L_vq)
            L_budget_list.append(L_budget)
            L_slt_list.append(L_slt)
            L_total_list.append(L_total)
    
    iterations = list(range(1, len(L_X_re_list) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.ylim(0,10)
    plt.plot(iterations, L_X_re_list, label='L_X_re')
    plt.plot(iterations, L_vq_list, label='L_vq')
    plt.plot(iterations, L_budget_list, label='L_budget')
    plt.plot(iterations, L_slt_list, label='L_slt')
    plt.plot(iterations, L_total_list, label='L_total')
    
    plt.title('DVQ-VAE Training Loss Components')
    plt.grid(True)
    plt.legend(loc='best')
    
    plot_dir = './data'
    plot_file = get_unique_path(os.path.join(plot_dir, 'DVQVAE_plot.png'))
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    

def main():
    loss_path, t2sgpt_loss_path = train_both_model(num_epochs=50, batch_size=5, codebook_size=64)
    plot_loss(loss_path)

if __name__ == "__main__":
    main()
