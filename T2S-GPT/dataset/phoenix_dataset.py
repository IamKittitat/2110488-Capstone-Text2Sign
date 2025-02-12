# T2s-GPT/dataset/pheonix_dataset.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignLanguageDataset(Dataset):
    def __init__(self, skel_file='T2S-GPT/data/sampledata/train.skels', text_file='T2S-GPT/data/sampledata/train.txt', vocab=None, seq_length=200, text_length=30, sign_language_dim=150, window_size=16):
        self.seq_length = seq_length
        self.text_length = text_length
        self.sign_language_dim = sign_language_dim
        self.window_size = window_size

        self.min_value = float("inf")
        self.max_value = -float("inf")

        with open(skel_file, 'r') as f:
            self.sign_language_data = f.readlines()
            self.sign_language_data = [line.strip().split(" ") for line in self.sign_language_data]
            self.sign_language_data = [[float(val) for val in line] for line in self.sign_language_data]
            # self.sign_language_data = [torch.tensor(line).reshape(-1, self.sign_language_dim + 1)[:, :-1] for line in self.sign_language_data]
            self.sign_language_data = [torch.tensor(line).reshape(-1, self.sign_language_dim) for line in self.sign_language_data]

        with open(text_file, 'r') as f:
            self.text_data = [line.strip().split() for line in f]

        if vocab:
            self.vocab = vocab
        else:
            self.vocab = self.build_vocab(self.text_data)


    def build_vocab(self, text_data):
        def yield_tokens(data):
            for text in data:
                yield text
        vocab = build_vocab_from_iterator(yield_tokens(text_data))
        return vocab
    
    def __len__(self):
        return len(self.sign_language_data)

    def __getitem__(self, idx):
        sign_language_sequence = self.sign_language_data[idx]

        spoken_language_text = self.text_data[idx]
        spoken_language_tensor = torch.tensor([self.vocab[word] for word in spoken_language_text[:self.text_length]], dtype=torch.long)

        return {
            'sign_language_sequence': sign_language_sequence.to(device),
            'spoken_language_text': spoken_language_tensor.to(device),
            'sign_language_original_length': sign_language_sequence.shape[0],
            'spoken_language_original_length': len(spoken_language_text)
        }