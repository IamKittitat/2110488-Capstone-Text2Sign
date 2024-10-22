# T2s-GPT/dataset/pheonix_dataset.py
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignLanguageDataset(Dataset):
    def __init__(self, skel_file='T2S-GPT/data/alldata/train.skels', text_file='T2S-GPT/data/alldata/train.txt', vocab=None, seq_length=200, text_length=30, sign_language_dim=150, window_size=16):
        self.seq_length = seq_length
        self.text_length = text_length
        self.sign_language_dim = sign_language_dim
        self.window_size = window_size

        self.min_value = float("inf")
        self.max_value = -float("inf")

        # Read and process skeleton data
        with open(skel_file, 'r') as f:
            self.sign_language_data = f.readlines()
            # Strip and split lines, then convert to float and reshape based on joint size
            self.sign_language_data = [line.strip().split(" ") for line in self.sign_language_data]
            self.sign_language_data = [[float(val) for val in line] for line in self.sign_language_data]
            self.sign_language_data = [torch.tensor(line).reshape(-1, self.sign_language_dim + 1)[:, :-1] for line in self.sign_language_data]

            # Normalize the skeleton data (calculate min and max values)
            for line in self.sign_language_data:
                self.min_value = min(self.min_value, line.min())
                self.max_value = max(self.max_value, line.max())

        # Normalize the skeleton data
        self.sign_language_data = [(line - self.min_value) / (self.max_value - self.min_value) for line in self.sign_language_data]

        # Read and process text data
        with open(text_file, 'r') as f:
            self.text_data = [line.strip().split() for line in f]

        # Build or use existing vocabulary for text data
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
        # Get skeleton sequence with window size
        start_index = torch.randint(0, self.sign_language_data[idx].shape[0] - self.window_size, (1,)).item()
        end_index = start_index + self.window_size
        sign_language_sequence = self.sign_language_data[idx][start_index:end_index]

        # Get the corresponding spoken language text and convert to indices
        spoken_language_text = self.text_data[idx]
        spoken_language_tensor = torch.tensor([self.vocab[word] for word in spoken_language_text[:self.text_length]], dtype=torch.long)

        return {
            'sign_language_sequence': sign_language_sequence.to(device),
            'spoken_language_text': spoken_language_tensor.to(device)
        }
