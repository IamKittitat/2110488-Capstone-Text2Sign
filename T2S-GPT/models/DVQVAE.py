# T2s-GPT/models/DVQVAE.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import PositionalEncoding

class DVQVAE_Encoder(nn.Module):
    def __init__(self, sign_language_dim, latent_dim, codebook_size, max_len=5000, dropout=0.1, num_layers = 6, decay=0.5):
        super(DVQVAE_Encoder, self).__init__()
        self.embedding = nn.Linear(sign_language_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()
        self.positional_encoding = PositionalEncoding(latent_dim, dropout, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=2048, batch_first=True), 
            num_layers = num_layers
        )
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        
        self.W2 = nn.Linear(latent_dim, latent_dim)
        self.W3 = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.threshold = 1.0

        # EMA parameters
        self.ema_decay = decay  # Exponential moving average decay factor
        self.ema_cluster_size = torch.zeros(codebook_size)  # Track codebook usage
        self.ema_embeddings = torch.zeros(codebook_size, latent_dim)  # Track EMA for embeddings

    def forward(self, X_T, is_training):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len , sign_language_dim]``
        """
        device = X_T.device
        self.ema_cluster_size = self.ema_cluster_size.to(device)
        self.ema_embeddings = self.ema_embeddings.to(device)
        self.codebook.weight.data = self.codebook.weight.data.to(device)

        X_T_prime = self.positional_encoding(self.relu(self.layer_norm(self.embedding(X_T))))
        H_T = self.transformer_encoder(X_T_prime)

        I_T = self.compute_information_weights(H_T).to(device) # Equation 3
        Z_T_l, D_T_l, S_T = self.downsample(H_T, I_T)  # Equation 4 and 5

        Z_quantized, codebook_indices = self.quantize(Z_T_l, is_training)
        return Z_quantized, D_T_l, S_T, Z_T_l, I_T, codebook_indices, H_T
    
    def compute_information_weights(self, H_T):
        H_residual = H_T
        H = self.relu(self.W2(H_T)) + H_residual  # Skip connection
        I_T = self.sigmoid(self.W3(H))  # I: [batch_size, seq_len, 1]
        I_T = I_T.squeeze(-1)  # I: [batch_size, seq_len]
        return I_T

    def downsample(self, H_T, I_T):
        device = H_T.device
        S_T = torch.cumsum(I_T, dim=1)//self.threshold
        Z_T_l = []
        D_T_l = []
        batch_size, seq_length, latent_dim = H_T.size()
        for t in range(int(torch.max(S_T).item()) + 1):
            Z_t = torch.zeros(batch_size, latent_dim, device=device)  # Initialize Z_t for the current segment
            D_t = torch.zeros(batch_size, device=device)
            for j in range(seq_length):
                F_j = (S_T[:,j] == t).int() # Equation 5 (condition on S_j)
                Z_t += H_T[:,j] * I_T[:,j].unsqueeze(1) * F_j.unsqueeze(1)  # Accumulate the sum
                D_t += F_j
            Z_T_l.append(Z_t)
            D_T_l.append(D_t)

        Z_T_l = torch.stack(Z_T_l, dim=1).to(device)
        D_T_l = torch.stack(D_T_l, dim=1).to(device).int() 
        return Z_T_l, D_T_l, S_T

    def quantize(self, z, is_training):
        device = z.device
        z_flattened = z.view(-1, z.size(-1)).to(device)
        self.ema_cluster_size = self.ema_cluster_size.to(device)
        self.ema_embeddings = self.ema_embeddings.to(device)
        codebook_indices = torch.argmin(torch.cdist(z_flattened, self.codebook.weight), dim=-1).view((z.shape[0], z.shape[1]))
        z_q = self.codebook(codebook_indices).view(z.size())

        # EMA Update
        if is_training:
            print("EMA")
            encoding_one_hot = torch.zeros(z_flattened.size(0), self.codebook_size, device=device)
            encoding_one_hot.scatter_(1, codebook_indices.view(-1, 1), 1)

            self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + (1 - self.ema_decay) * encoding_one_hot.sum(0)
            dw = torch.matmul(encoding_one_hot.t(), z_flattened)
            self.ema_embeddings = self.ema_decay * self.ema_embeddings + (1 - self.ema_decay) * dw

            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (self.ema_cluster_size) / (n + self.codebook_size * 1e-5) * n
            self.codebook.weight.data.copy_(self.ema_embeddings / self.ema_cluster_size.unsqueeze(1))
            
            # Codebook Reset
            inactive_codes = (self.ema_cluster_size < 1e-5).nonzero()
            if inactive_codes.numel() > 0:
                self.codebook.weight.data[inactive_codes] = torch.randn_like(self.codebook.weight[inactive_codes])

        return z_q, codebook_indices
    
    def get_codebook(self, S):
        return self.codebook(S)

class DVQVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, sign_language_dim = 150, max_len=5000, dropout=0.1, num_layers = 6):
        super(DVQVAE_Decoder, self).__init__()
        self.positional_encoding = PositionalEncoding(latent_dim, dropout, max_len) 
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=2048, batch_first=True), 
            num_layers= num_layers
        )
        self.embedding = nn.Linear(latent_dim, sign_language_dim)

    def forward(self, Z_quantized, D_T_l, H_T):
        X_hat = self.positional_encoding(self.length_regulator(Z_quantized, D_T_l))
        X_re = self.embedding(self.transformer_decoder(X_hat, H_T))
        return X_re

    def generate(self, Z_quantized, D_T_l):
        X_hat = self.positional_encoding(self.length_regulator(Z_quantized, D_T_l))
        X_re = self.transformer_decoder(X_hat, X_hat)
        return X_re

    def length_regulator(self, z_q, durations):
        expanded_list = []

        for b in range(z_q.shape[0]):
            expanded_seq = z_q[b].repeat_interleave(durations[b], dim=0) 
            expanded_list.append(expanded_seq)
        return torch.stack(expanded_list)  


class DVQVAELoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=1.0, R=12):
        super(DVQVAELoss, self).__init__()
        self.lambda1 = lambda1  # Coefficient for the commitment loss
        self.lambda2 = lambda2  # Coefficient for the budget loss
        self.lambda3 = lambda3  # Coefficient for the sign language translation loss
        self.R = R  # Expected downsampling rate

    def smooth_l1_loss(self, pred, target):
        return F.smooth_l1_loss(pred, target, reduction='mean')

    def velocity(self, x):
        v = x[:, 1:, :] - x[:, :-1, :]
        v_full = torch.zeros_like(x)
        v_full[:, :-1, :] = v
        v_full[:, -1, :] = x[:, -1, :]
        return v_full

    def forward(self, X_T, X_re, Z_T_l, Z_quantized, I_T, T, P_Y_given_X_re, loss_path = None):
        # Reconstruction Loss (Eq. 8)
        L_X_re = self.smooth_l1_loss(X_T, X_re)
        L_re = L_X_re + self.smooth_l1_loss(self.velocity(X_T), self.velocity(X_re))
    
        # Embedding Loss & Commitment Loss & vq Loss (Eq. 7)
        L_embed = torch.mean((Z_T_l - Z_quantized.detach()) ** 2)
        L_commit = self.lambda1 * torch.mean((Z_T_l.detach() - Z_quantized) ** 2)
        L_vq = L_re + L_embed + L_commit

        # Budget Loss (Eq. 9)
        L_budget = torch.mean(torch.clamp(torch.sum(I_T) - T / self.R, min=0))

        # Sign Language Translation Auxiliary Loss (Eq. 10)
        # L_slt = -torch.mean(torch.log(P_Y_given_X_re + 1e-8))

        # Final Loss (Eq. 11)
        # L_total = L_vq + self.lambda2 * L_budget + self.lambda3 * L_slt
        L_total = L_vq + self.lambda2 * L_budget

        # Append loss into file
        if(loss_path is not None):
            with open(loss_path, "a") as f:
                f.write(f"{L_X_re},{L_vq},{L_budget},{L_total}\n")
                # f.write(f"{L_X_re},{L_vq},{L_budget},{L_slt},{L_total}\n")

        return L_total
