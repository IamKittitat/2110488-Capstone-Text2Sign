import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

class T2S_GPT(nn.Module):
    def __init__(self, vocab_size, codebook_size, max_duration, device, embed_dim = 1024, nhead = 16, dim_feedforward = 4096, dropout=0.1):
        super(T2S_GPT, self).__init__()
        
        self.device = device
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
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

        self.code_output_layer = nn.Linear(embed_dim, codebook_size)
        self.duration_output_layer = nn.Linear(embed_dim, max_duration + 1)
    
    def forward(self, Y, S_T, D_T_l, SOS_TOKEN = 11, EOS_TOKEN = 12):
        start_token = torch.tensor([SOS_TOKEN]).repeat(S_T.size(0), 1)
        end_token = torch.tensor([EOS_TOKEN]).repeat(S_T.size(0), 1)
        S_T_with_start = torch.cat((start_token, S_T, end_token), dim=1)
        D_T_l_with_start = torch.cat((start_token, D_T_l, end_token), dim=1)

        Y_embed = self.text_embedding(Y)
        S_T_embed = self.code_embedding(S_T_with_start[:, :-1])
        D_T_l_embed = self.duration_embedding(D_T_l_with_start[:, :-1])

        # Code Transformer: Y, S_T -> S_T_pred
        tgt_mask = self.get_tgt_mask(S_T_embed.size(1)).to(self.device)
        H_code = self.code_transformer(Y_embed, S_T_embed, tgt_mask=tgt_mask)
        code_transformer_logits = self.code_output_layer(H_code)
        S_T_pred = torch.argmax(F.softmax(code_transformer_logits, dim=2), dim=2)

        # Duration Transformer: D_T_l, S_T -> D_T_l_pred ??
        tgt_mask_duration = self.get_tgt_mask(D_T_l_embed.size(1)).to(self.device)
        # TODO: Recheck this (EQ. 13)
        H_dur = self.duration_transformer(H_code, D_T_l_embed, tgt_mask=tgt_mask_duration)
        duration_transformer_logits = self.duration_output_layer(H_dur)
        D_T_l_pred = torch.argmax(F.softmax(duration_transformer_logits, dim=2), dim=2)

        return S_T_pred, S_T_with_start[:, 1:], code_transformer_logits, D_T_l_pred, D_T_l_with_start[:, 1:], duration_transformer_logits


    def generate(self, Y, max_length, SOS_TOKEN=11, EOS_TOKEN=12):
        self.eval()
        with torch.no_grad():
            Y_embed = self.text_embedding(Y)

            S_T = torch.full((Y.size(0), 1), SOS_TOKEN, dtype=torch.long, device=self.device)
            D_T_l = torch.full((Y.size(0), 1), SOS_TOKEN, dtype=torch.long, device=self.device)  

            new_token = S_T
            new_duration = D_T_l

            for _ in range(max_length):
                S_T_embed = self.code_embedding(new_token)
                H_code = self.code_transformer(Y_embed, S_T_embed)
                code_transformer_logits = nn.Linear(self.embed_dim, self.embed_dim)(H_code)
                next_token = torch.argmax(nn.Softmax(dim=2)(code_transformer_logits), dim=2, keepdim=True).squeeze(dim=2)
                S_T = torch.cat((S_T, next_token), dim=1)
                
                D_T_l_embed = self.duration_embedding(new_duration)
                output_dur = self.duration_transformer(H_code, D_T_l_embed)
                duration_transformer_logits = nn.Linear(self.embed_dim, self.embed_dim)(output_dur)
                next_duration = torch.argmax(nn.Softmax(dim=2)(duration_transformer_logits), dim=2, keepdim=True).squeeze(dim=2)
                D_T_l = torch.cat((D_T_l, next_duration), dim=1)

                if ((next_token == EOS_TOKEN).all() | (next_duration == EOS_TOKEN).all()):
                    break

        return S_T, D_T_l


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

    def forward(self, code_transformer_logits, S_T_expected, D_T_l_pred, D_T_l_expected, loss_path = None):
        L_code = F.cross_entropy(code_transformer_logits.view(-1, code_transformer_logits.size(-1)), S_T_expected.reshape(-1))
        print("D_T_l", D_T_l_pred.shape, D_T_l_expected.shape)
        L_duration = F.mse_loss(D_T_l_pred, D_T_l_expected.float())

        L_total = L_code + L_duration

        # Append loss into file
        if(loss_path is not None):
            with open(loss_path, "a") as f:
                f.write(f"{L_code},{L_duration},{L_total}\n")

        return L_total