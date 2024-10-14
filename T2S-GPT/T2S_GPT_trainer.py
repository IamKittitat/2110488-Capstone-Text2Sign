import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.random_dataset import RandomDataset
from models.T2S_GPT import T2S_GPT, T2SGPTLoss
from models.DVQVAE import DVQVAE_Encoder

def train_t2s_gpt_model(epochs=10, batch_size=32, learning_rate=1e-4,
                         T=100, vocab_size=500, codebook_size=1024,
                         sign_language_dim=512, latent_dim=512):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size).to(device)
    model = T2S_GPT(vocab_size=vocab_size, codebook_size=codebook_size, max_duration=100, device=device).to(device)
    t2sgpt_loss = T2SGPTLoss()
    dataset = RandomDataset(T, sign_language_dim, sign_language_dim, vocab_size, num_samples=30)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, total_code_loss, total_duration_loss = 0, 0, 0
        
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence
            
            # Forward pass through the encoder
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T)
            
            # Forward pass through the model
            optimizer.zero_grad()
            S_T_pred, S_T_expected, H_code, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits = model(Y, codebook_indices, D_T_l)
            
            # Compute loss
            loss_total, loss_code, loss_duration = t2sgpt_loss(code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected)
            
            # Backward pass
            loss_total.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss_total.item()
            total_code_loss += loss_code.item()
            total_duration_loss += loss_duration.item()
        
        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}, Code Loss: {total_code_loss:.4f}, Duration Loss: {total_duration_loss:.4f}")

def main():
    train_t2s_gpt_model(epochs=10, batch_size=32, learning_rate=1e-4)

if __name__ == "__main__":
    main()
