# T2s-GPT/DVQVAE_rechecker.py
import torch
from models.DVQVAE import DVQVAE_Encoder, DVQVAE_Decoder, DVQVAELoss

sign_language_dim = 150
T = 100
batch_size = 32
latent_dim = 512
vocab_size = 500
codebook_size = 1024
output_dim = sign_language_dim
num_epochs = 20
text_length = 10

encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size)
decoder = DVQVAE_Decoder(latent_dim, output_dim)
loss_fn = DVQVAELoss(lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12)

X_T = torch.randn(batch_size, T, sign_language_dim)  # batch_size, sequence_length, sign_language_dim
Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T, is_training = True)
print(X_T.shape, Z_quantized.shape, D_T_l.shape, S_T.shape, Z_T_l.shape, I_T.shape, codebook_indices.shape, H_T.shape)
X_re = decoder(Z_quantized, D_T_l, H_T)
print("X_Re", X_re.shape)
P_Y_given_X_re = torch.rand(batch_size)
loss = loss_fn(X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re)

print(f"Z_quantized: {Z_quantized.shape}, D_T_l: {D_T_l.shape}, S_T: {S_T.shape}, \nZ_T_l: {Z_T_l.shape}, I_T: {I_T.shape}, codebook_indices: {codebook_indices.shape}, H_T: {H_T.shape}")
print(f"X_re: {X_re.shape}")