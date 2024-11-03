# T2s-GPT/T2S_GPT_trainer.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.random_dataset import RandomDataset
from dataset.phoenix_dataset import SignLanguageDataset
from models.T2S_GPT import T2S_GPT, T2SGPTLoss
from models.DVQVAE import DVQVAE_Encoder
from utils.file_utils import get_unique_path
from utils.pad_seq import pad_collate_fn

def train_t2s_gpt_model(epochs=10, batch_size=32, learning_rate=1e-4,
                         T=100, vocab_size=500, codebook_size=1024,
                         sign_language_dim=512, latent_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size).to(device)
    model = T2S_GPT(vocab_size=vocab_size, codebook_size=codebook_size, max_duration=100, device=device).to(device)
    t2sgpt_loss = T2SGPTLoss()
    ## RandomDataset
    dataset = RandomDataset(T, sign_language_dim, sign_language_dim, vocab_size, num_samples=5)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ## SignLanguageDataset
    # dataset = SignLanguageDataset()
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Load the checkpoint
    checkpoint = torch.load('./trained_model/dvqvae_model_1.pth')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    loss_path = get_unique_path('./data/T2SGPT_loss.txt')

    for epoch in range(epochs):
        model.train()
        total_loss, total_code_loss, total_duration_loss = 0, 0, 0
        
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence
            
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=False)
            S_T_pred, S_T_expected, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits = model(Y, codebook_indices, D_T_l)
            loss_total = t2sgpt_loss(code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected, loss_path)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}")
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, './trained_model/t2sgpt_model_1.pth')

    # print("SHAPE:",D_T_l_pred.shape, D_T_l_expected.shape)
    # print(D_T_l)
    # print(D_T_l_pred)
    # print(D_T_l_expected)
    return total_loss
    

def main():
    train_t2s_gpt_model(epochs=3, batch_size=32, learning_rate=1e-5, codebook_size = 64, sign_language_dim=150)

if __name__ == "__main__":
    main()
