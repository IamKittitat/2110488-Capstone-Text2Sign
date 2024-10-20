import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

class T2S_GPT(nn.Module):
    def __init__(self, vocab_size, codebook_size, max_duration, device, embed_dim = 1024, nhead = 16, dim_feedforward = 4096, dropout=0.1):
        super(T2S_GPT, self).__init__()
        
        self.device = device
        self.text_embedding = nn.Embedding(vocab_size, embed_dim) 
        self.code_embedding = nn.Embedding(codebook_size, embed_dim)
        self.duration_embedding = nn.Embedding(max_duration + 1, embed_dim)

        self.code_transformer = Transformer(
            d_model=embed_dim, 
            nhead=nhead, 
            num_decoder_layers=18,
            num_encoder_layers=18,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
        )

        self.duration_transformer = Transformer(
            d_model=embed_dim, 
            nhead=nhead, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
        )
    
    def forward(self, Y, S_T, D_T_l, embed_dim = 1024):
        # Code Transformer: Y, S_T -> S_T_pred
        # TODO: Do something with SOS_TOKEN, EOS_TOKEN?
        start_token = torch.tensor([11]).repeat(S_T.size(0), 1)
        S_T_with_start = torch.cat((start_token, S_T), dim=1)
        D_T_l_with_start = torch.cat((start_token, D_T_l), dim=1)

        Y_embed = self.text_embedding(Y)
        S_T_embed = self.code_embedding(S_T_with_start)
        S_T_input = S_T_embed[:, :-1]
        S_T_expected = S_T_with_start[:, 1:]
        sequence_length = S_T_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)
        
        H_code = self.code_transformer(Y_embed, S_T_input, tgt_mask=tgt_mask)
        code_transformer_logits = nn.Linear(embed_dim, embed_dim)(H_code)
        S_T_pred = torch.argmax(nn.Softmax(dim=2)(code_transformer_logits), dim = 2)
        # Duration Transformer: D_T_l, S_T -> D_T_l_pred ??
        D_T_l_embed = self.duration_embedding(D_T_l_with_start)
        D_T_l_input = D_T_l_embed[:, :-1]
        D_T_l_expected = D_T_l_with_start[:, 1:]
        sequence_length = D_T_l_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)
        # TODO: Recheck this
        Ny = Y_embed.size(1)
        l = S_T_input.size(1)//D_T_l_input.size(1)
        # print(f"S_T_pred: {S_T_pred.shape}, S_T_expected: {S_T_expected.shape}, H_code: {H_code.shape}")
        # print(f"H_code[:, Ny: Ny + l - 1]: {H_code[:, Ny: Ny + l - 1].shape}, S_T_pred[:, : l + 1]: {S_T_pred[:, : l + 1].shape}")
        # H_dur = H_code[:, Ny: Ny + l - 1] + S_T_pred[:, : l + 1]
        H_dur = H_code
        duration_transformer_logits = self.duration_transformer(H_dur, D_T_l_input, tgt_mask=tgt_mask)
        D_T_l_pred = torch.argmax(nn.Softmax(dim=2)(nn.Linear(embed_dim, D_T_l.shape[-1])(duration_transformer_logits)), dim = 2)
        # print(f"D_T_l_embed: {D_T_l_embed.shape}, D_T_l_input: {D_T_l_input.shape}, D_T_l_expected: {D_T_l_expected.shape}, D_T_l_pred: {D_T_l_pred.shape}")

        return S_T_pred, S_T_expected, H_code, code_transformer_logits, D_T_l_pred, D_T_l_expected, duration_transformer_logits

    
    def get_tgt_mask(self, size):
        # Generates a squere matrix where each row allows words to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int):
        return (matrix == pad_token)
    

class T2SGPTLoss(nn.Module):
    def __init__(self):
        super(T2SGPTLoss, self).__init__()

    def forward(self, code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected, loss_path):
        L_code = F.cross_entropy(code_transformer_logits.view(-1, code_transformer_logits.size(-1)), S_T_expected.reshape(-1))

        print("D_T_l", D_T_l_pred.shape, D_T_l_expected.shape)
        L_duration = F.mse_loss(D_T_l_pred, D_T_l_expected.float())

        L_total = L_code + L_duration

        # Append loss into file
        with open(loss_path, "a") as f:
            f.write(f"{L_code},{L_duration},{L_total}\n")

        return L_total