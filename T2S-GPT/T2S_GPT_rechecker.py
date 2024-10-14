import torch
from models.T2S_GPT import T2S_GPT, T2SGPTLoss
from models.DVQVAE import DVQVAE_Encoder

T = 100
vocab_size = 500
codebook_size = 1024
d_model = 512
batch_size = 32
sign_language_dim = 512
latent_dim = 512
output_dim = sign_language_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T2S_GPT(vocab_size=vocab_size, codebook_size=codebook_size, max_duration = 100, device = device)
t2sgpt_loss = T2SGPTLoss()
encoder = DVQVAE_Encoder(sign_language_dim, latent_dim, codebook_size)


X_T = torch.randn(batch_size, T, sign_language_dim)  # batch_size, sequence_length, sign_language_dim
Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T = encoder(X_T)
Y = torch.randint(0, vocab_size, (batch_size, 10))
print(f"Y: {Y.shape}, codebook_indices: {codebook_indices.shape}, D_T_l: {D_T_l.shape}")
S_T_pred, S_T_expected, H_code, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits = model(Y, codebook_indices, D_T_l)
loss = t2sgpt_loss(code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected)
print(f"Loss: {loss}")

