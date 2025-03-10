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
from utils.visualization import plot_loss
from utils.pad_seq import pad_collate_fn

def min_max_scale(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)  # Avoid division by zero
    return scaled_tensor, tensor_min, tensor_max

def train_t2s_gpt_model(dvq_path, epochs=10, batch_size=32, learning_rate=1e-4,
                         T=100, vocab_size=500, codebook_size=1024,
                         sign_language_dim=150, latent_dim=512):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size).to(device)
    model = T2S_GPT(vocab_size=vocab_size, codebook_size=codebook_size, max_duration=400, device=device).to(device)
    t2sgpt_loss = T2SGPTLoss()
    
    ## RandomDataset
    # dataset = RandomDataset(T, sign_language_dim, sign_language_dim, vocab_size, num_samples=5)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ## SignLanguageDataset
    # Relative Angle Data
    # skel_file = os.path.join(current_dir, 'data/sampledata_relative/train.skels')
    # text_file = os.path.join(current_dir, 'data/sampledata_relative/train.txt')
    # Absolute Position Data
    # skel_file = os.path.join(current_dir, 'data/sampledata/train.skels')
    # text_file = os.path.join(current_dir, 'data/sampledata/train.txt')
    skel_file = os.path.join(current_dir, 'data/scaled_skeleton/dev.skels')
    text_file = os.path.join(current_dir, 'data/scaled_skeleton/dev.txt')

    dataset = SignLanguageDataset(skel_file=skel_file, text_file=text_file, sign_language_dim = sign_language_dim, window_size=394)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Load the checkpoint
    folder_dir = get_unique_path(os.path.join(current_dir, 'result/T2SGPT_trainer'))
    os.makedirs(folder_dir, exist_ok=True)
    loss_path = os.path.join(folder_dir, 'T2SGPT_loss.txt')
    model_path = os.path.join(folder_dir, 'T2SGPT_model.pth')

    checkpoint = torch.load(os.path.join(current_dir, f'result/{dvq_path}/DVQVAE_model.pth'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    for epoch in range(epochs):
        model.train()
        total_loss, total_code_loss, total_duration_loss = 0, 0, 0
        
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence
            # X_original_length = batch['sign_language_original_length']
            # Y_original_length = batch['spoken_language_original_length']

            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=False)
            
            S_T_pred, S_T_expected, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits = model(Y, codebook_indices, D_T_l)
            print("CHECK", S_T_pred.shape, S_T_expected.shape, D_T_l_pred.shape, D_T_l_expected.shape)
            print(S_T_pred, S_T_expected, D_T_l_pred, D_T_l_expected)
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
    }, model_path)

    return total_loss, folder_dir
    
def main():
    total_loss, folder_dir = train_t2s_gpt_model(dvq_path = "DVQVAE_trainer_4", epochs=1, batch_size=4, learning_rate=1e-5, codebook_size = 64, sign_language_dim=1659)
    loss_path = os.path.join(folder_dir, 'T2SGPT_loss.txt')
    save_path = os.path.join(folder_dir, 'T2SGPT_plot.png')
    plot_loss(loss_path, ["L_code", "L_duration", "L_total"], "T2S-GPT Training Loss", save_path, y_lim = 10000)
    print("Complete training, Save as",save_path)

if __name__ == "__main__":
    main()
