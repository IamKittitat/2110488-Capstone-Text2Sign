# T2s-GPT/utils/pad_seq.py
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    # sign_language_sequences = [item['sign_language_sequence'] for item in batch]
    # spoken_language_texts = [item['spoken_language_text'] for item in batch]

    # return {
    #     'sign_language_sequence': sign_language_sequences,
    #     'spoken_language_text': spoken_language_texts
    # }


    # Extract sign language sequences and text sequences from the batch
    sign_language_sequences = [item['sign_language_sequence'] for item in batch]
    spoken_language_texts = [item['spoken_language_text'] for item in batch]
    sign_language_original_length = [item['sign_language_original_length'] for item in batch]
    spoken_language_original_length = [item['spoken_language_original_length'] for item in batch]

    # Pad sign language sequences and text sequences
    X_T_padded = pad_sequence(sign_language_sequences, batch_first=True)  # Pads sign language sequences
    Y_padded = pad_sequence(spoken_language_texts, batch_first=True)      # Pads text sequences

    return {
        'sign_language_sequence': X_T_padded,
        'spoken_language_text': Y_padded,
        'sign_language_original_length' : sign_language_original_length,
        'spoken_language_original_length' : spoken_language_original_length
    }