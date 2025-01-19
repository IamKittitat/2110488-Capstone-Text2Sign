import numpy as np
from scipy import linalg

def n_grams(text, n):
    tokens = text.split()
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def calculate_bleu(reference_texts, predicted_text, max_order):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_texts: list of lists of references for each translation.
    predicted_text: list of translations to score.
    max_order: Maximum n-gram order to use when computing BLEU score.

  Returns:
    BLEU score.
  """
    reference_ngrams = [set(n_grams(ref, max_order)) for ref in reference_texts]
    predicted_ngrams = n_grams(predicted_text, max_order)

    matches = 0
    for ngram in predicted_ngrams:
        if any(ngram in ref for ref in reference_ngrams):
            matches += 1

    precision = matches / len(predicted_ngrams) if predicted_ngrams else 0

    ref_lengths = [len(ref.split()) for ref in reference_texts]
    pred_length = len(predicted_text.split())
    closest_ref_length = min(ref_lengths, key=lambda ref_len: abs(ref_len - pred_length))
    brevity_penalty = 1 if pred_length > closest_ref_length else pow(2.71828, 1 - closest_ref_length / pred_length)

    bleu_score = brevity_penalty * precision
    
    return bleu_score

def calculate_fid(real_features, generated_features, eps=1e-6):
    """
    Calculate FID loss between two sets of features.
    
    Args:
        real_features (numpy.ndarray): Feature vectors from real data, shape (N, D).
        generated_features (numpy.ndarray): Feature vectors from generated data, shape (N, D).

    Returns:
        float: FID loss value.
    """
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_generated = np.cov(generated_features, rowvar=False)

    mean_diff = np.sum((mu_real - mu_generated) ** 2)
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_generated), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_generated + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (mean_diff + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * tr_covmean)