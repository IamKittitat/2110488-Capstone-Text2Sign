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

def predict(Y, dvqvae_path, t2sgpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DVQVAE_Encoder(sign_language_dim=512, latent_dim=512, codebook_size=64).to(device).eval()
    decoder = DVQVAE_Decoder(latent_dim = 512, output_dim = 512).to(device).eval()
    t2sgpt = T2S_GPT(vocab_size=500, codebook_size=64, embed_dim=64, max_duration=100, device=device).to(device).eval()

    dvqvae_checkpoint = torch.load(dvqvae_path)
    t2sgpt_checkpoint = torch.load(t2sgpt_path)
    decoder.load_state_dict(dvqvae_checkpoint['decoder_state_dict'])
    t2sgpt.load_state_dict(t2sgpt_checkpoint['model_state_dict'])

    S_T_pred, D_T_pred = t2sgpt.generate(Y, max_length=5, SOS_TOKEN=11, EOS_TOKEN=12)
    Z_quantized = encoder.get_codebook(S_T_pred)
    X_re = decoder.generate(Z_quantized, D_T_pred)
    print(f"S_T_pred: {S_T_pred.shape}, D_T_pred: {D_T_pred.shape}, Z_quantized: {Z_quantized.shape}, X_re: {X_re.shape}")
    print(D_T_pred)
    return X_re


def main():
    # Generate Random Tensor (batch_size, length)
    Y = torch.randint(0, 500, (5, 100))
    X_re = predict(Y, "./trained_model/dvqvae_model_2.pth", "./trained_model/t2sgpt_model_2.pth")
    print(X_re)

if __name__ == "__main__":
    main()
