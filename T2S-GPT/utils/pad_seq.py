from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    sign_language_sequences = [item['sign_language_sequence'] for item in batch]
    spoken_language_texts = [item['spoken_language_text'] for item in batch]
    sign_language_original_length = [item['sign_language_original_length'] for item in batch]
    spoken_language_original_length = [item['spoken_language_original_length'] for item in batch]

    X_T_padded = pad_sequence(sign_language_sequences, batch_first=True)
    Y_padded = pad_sequence(spoken_language_texts, batch_first=True)

    return {
        'sign_language_sequence': X_T_padded,
        'spoken_language_text': Y_padded,
        'sign_language_original_length' : sign_language_original_length,
        'spoken_language_original_length' : spoken_language_original_length
    }