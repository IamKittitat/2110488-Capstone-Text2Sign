import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, seq_length, sign_language_dim, output_dim, vocab_size, num_samples=300, text_length=10):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.sign_language_dim = sign_language_dim
        self.text_length = text_length
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        
        self.data = []
        for _ in range(self.num_samples):
            sign_language_sequence = torch.rand(self.seq_length, self.sign_language_dim)
            spoken_language_text = torch.randint(0, self.vocab_size, (self.text_length,))
            # torch.randint(0, vocab_size, (batch_size, 10))
            self.data.append({
                'sign_language_sequence': sign_language_sequence,
                'spoken_language_text': spoken_language_text
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]