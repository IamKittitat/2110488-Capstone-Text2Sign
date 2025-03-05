# T2s-GPT/DVQVAE_trainer.py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna

from models.DVQVAE import DVQVAE_Encoder, DVQVAE_Decoder, DVQVAELoss
from dataset.random_dataset import RandomDataset
from dataset.phoenix_dataset import SignLanguageDataset
from utils.file_utils import get_unique_path
from utils.visualization import plot_loss
from utils.pad_seq import pad_collate_fn

def train_dvqvae_model(num_epochs=500, batch_size=32, sign_language_dim=512,
                       T=100, latent_dim=512, vocab_size=500, codebook_size=1024, 
                       output_dim=512):
    # Prepare dataset and data loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    ## SignLanguageDataset
    skel_file = os.path.join(current_dir, 'data/thsl_skeleton/train.skels')
    text_file = os.path.join(current_dir, 'data/thsl_skeleton/train.txt')

    dataset = SignLanguageDataset(skel_file=skel_file, text_file=text_file, sign_language_dim = sign_language_dim, window_size=394)
    # TODO : FIX LATER TO HAVE VAL_LOADER
    # total_size = len(dataset)
    # train_size = int(0.7 * total_size)
    # val_size = total_size - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)


    def objective(trial):
        """Hyperparameter tuning objective function for learning rate."""
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # Tune learning rate
        beta1 = trial.suggest_float('beta1', 0.8, 0.95)  # Tune beta1
        beta2 = trial.suggest_float('beta2', 0.98, 0.9999)  # Tune beta2
        t_max = 50

        # Initialize models, loss function, optimizer, and scheduler
        encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size).to(device)
        decoder = DVQVAE_Decoder(latent_dim, output_dim, sign_language_dim=sign_language_dim).to(device)
        loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12).to(device)

        optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, betas=(beta1, beta2))
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)  # Tuning over 20 epochs

        total_train_loss = 0.0
        for epoch in range(t_max):  # Short training for hyperparameter tuning
            encoder.train()
            decoder.train()
            for batch in train_loader:
                X_T = batch['sign_language_sequence'].to(device)
                Y = batch['spoken_language_text'].to(device)
                X_original_length = batch['sign_language_original_length']

                Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=True)
                X_re = decoder(Z_quantized, D_T_l, H_T)

                P_Y_given_X_re = torch.ones(batch_size, output_dim, T, device=device)
                train_loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()

            scheduler.step()

        return total_train_loss / len(train_loader)  # Minimize loss
    
    # Run Optuna Hyperparameter Optimization**
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_lr = study.best_params['lr']
    best_beta1 = study.best_params['beta1']
    best_beta2 = study.best_params['beta2']
    print(f"Best Hyperparameters - LR: {best_lr}, Beta1: {best_beta1}, Beta2: {best_beta2}")
    
    # Initialize models, loss function, optimizer, and scheduler
    encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size).to(device)
    decoder = DVQVAE_Decoder(latent_dim, output_dim, sign_language_dim = sign_language_dim).to(device)
    loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12).to(device)

    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=best_lr, betas=(best_beta1, best_beta2))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_loss_list = []

    folder_dir = get_unique_path(os.path.join(current_dir, 'result/DVQVAE_trainer'))
    os.makedirs(folder_dir, exist_ok=True)
    loss_path = os.path.join(folder_dir, 'DVQVAE_loss.txt')
    model_path = os.path.join(folder_dir, 'DVQVAE_model.pth')

    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        total_train_loss = 0.0
        for batch in train_loader:
            X_T = batch['sign_language_sequence'].to(device)  # Input: sign language sequence
            Y = batch['spoken_language_text'].to(device)       # Target: text sequence
            X_original_length = batch['sign_language_original_length']
            # Y_original_length = batch['spoken_language_original_length']

            # Forward pass through DVQ-VAE
            Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training=True)
            if(epoch % 50 == 0):
                print("Epoch:", epoch)
                print(codebook_indices)
            X_re = decoder(Z_quantized, D_T_l, H_T)

            # For demonstration, replace this with actual values
            P_Y_given_X_re = torch.ones(batch_size, output_dim, T, device=device)
            # print("CHECKER", X_T.shape, X_re.shape, Z_T_l.shape, Z_quantized.shape, I_T.shape, T, P_Y_given_X_re.shape)
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

    return train_loss_list, X_T.detach().cpu().numpy(), X_re.detach().cpu().numpy(), X_original_length, folder_dir

def main():
    loss_list, X_T, X_re, X_original_length, folder_dir = train_dvqvae_model(num_epochs=500, batch_size=4, sign_language_dim=1659, T=100, latent_dim=512, 
                                                          vocab_size=500, codebook_size=64,  output_dim=1659)
    loss_path = os.path.join(folder_dir, 'DVQVAE_loss.txt')
    save_path = os.path.join(folder_dir, 'DVQVAE_plot.png')
    plot_loss(loss_path, ["L_X_re", "L_vq", "L_budget", "L_total"], "DVQ-VAE Training Loss", save_path)
    print("Complete training, Save as",save_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    global_min = np.load(os.path.join(current_dir, "./constant/preproces_data_global_min.npy"))
    global_max = np.load(os.path.join(current_dir, "./constant/preproces_data_global_max.npy"))

    with open(os.path.join(folder_dir, 'X_re.skels'), "w") as f:
        for X_T_i, X_re_i, X_original_length_i in zip(X_T, X_re, X_original_length):
            print(X_T_i.shape, X_re_i.shape, X_original_length_i)
            X_re_i_unpadded = X_re_i[0:X_original_length_i, : ]
            X_unscaled_i = (X_re_i_unpadded + 1)*(global_max - global_min)/2 + global_min
            f.write(' '.join(map(str, X_unscaled_i.flatten())) + '\n')

if __name__ == "__main__":
    main()
