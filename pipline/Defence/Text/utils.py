import nltk
import numpy as np
import random
from nltk.tokenize import word_tokenize
from transformers import pipeline

def mask_and_demask(
    filtered_dataset_text,
    tokenizer,
    fill_mask,
    num_voter=5,   
    verbose=True,  
    mask_pct=0.2,  
    pos_weights=None,
    use_random_mask=False 
):
    """
    Perform masking and unmasking operations on input text. 
    Supports two modes: "per-word replacement" or "random masking".

    Parameters:
    - filtered_dataset_text: List of input texts.
    - tokenizer: Tokenizer object.
    - fill_mask: Model used for mask filling (e.g., BERT's fill-mask pipeline).
    - num_voter: Number of variants to generate per input via random masking.
    - verbose: Whether to print detailed generation information.
    - mask_pct: Percentage of words to mask in random mask mode.
    - pos_weights: (Optional) Used to adjust part-of-speech weighting for masking, 
                   e.g., {'NN': 2.0} makes nouns more likely to be masked.
    - use_random_mask: Whether to use random masking (True for random, False for per-word).

    Returns:
    - List[str]: List of newly generated sentences.
    """

    modified_adv_texts = []  # Store all generated new sentences

    for i, example in enumerate(filtered_dataset_text):
        
        # 1) Tokenize the input text using nltk
        tokens = word_tokenize(example)
        # Replace the last token with '?'
        tokens[-1] = '?'

        # =========================
        # Random Masking Mode
        # =========================
        if use_random_mask:

            # a) Get POS (Part-of-Speech) tags; if pos_weights is provided, use it to influence sampling probabilities
            pos_tags = nltk.pos_tag(tokens)  # [(word, pos), (word, pos), ...]
            weights = np.ones(len(tokens), dtype=float)  # Initial weight is 1 for all tokens
            if pos_weights is not None:
                for idx, (_, pos) in enumerate(pos_tags):
                    if pos in pos_weights:
                        weights[idx] = pos_weights[pos]
            # Normalize weights so the sum is 1
            weights /= weights.sum()  

            # b) Duplicate tokens num_voter times: shape = (num_voter, len(tokens))
            tokens_array = np.tile(tokens, (num_voter, 1))

            # c) Perform random masking for each copy
            #    Calculate the number of tokens to mask
            num_tokens_to_mask = max(2, int(len(tokens) * mask_pct))
            #    For each row, randomly select positions to mask
            mask_indices_list = [
                np.random.choice(
                    range(len(tokens)), 
                    size=num_tokens_to_mask, 
                    replace=False, 
                    p=weights
                )
                for _ in range(num_voter)
            ]

            # d) Replace selected tokens with [MASK]
            tokens_array = tokens_array.astype(object)
            tokens_array[np.arange(num_voter)[:, None], mask_indices_list] = "[MASK]"

            # e) Convert each row of tokens to a string -> masked_sentences
            v_convert_tokens_to_string = np.vectorize(tokenizer.convert_tokens_to_string, signature='(n)->()', otypes=[object])
            masked_sentences = v_convert_tokens_to_string(tokens_array).reshape(-1, 1)

            # Get indices of masked tokens
            replace_idxs = np.argwhere(tokens_array == '[MASK]')
            # Unmask the texts using fill_mask and collect results
            unmasked_text_suggestions = fill_mask([list(masked_sentence) for masked_sentence in masked_sentences], top_k=1)
            replacement_tokens = [token_info[0]['token_str'] for sentence in unmasked_text_suggestions for token_info in sentence]
            tokens_array[replace_idxs[:, 0], replace_idxs[:, 1]] = replacement_tokens
            unmasked = v_convert_tokens_to_string(tokens_array).reshape(-1,)
            [modified_adv_texts.append(unmasked[i]) for i in range(num_voter)]
            
            if verbose:
                print(modified_adv_texts)

        # =========================
        # Per-Word Replacement Mode
        # =========================
        else:
            # Replace each token in the sentence one by one with [MASK], then use fill_mask to replace it
            for idx in range(len(tokens)):
                temp_tokens = tokens.copy()
                temp_tokens[idx] = "[MASK]"

                text_str = " ".join(temp_tokens)
                results = fill_mask(text_str, top_k=1)  # Only use top prediction
                if len(results) > 0:
                    predicted_token = results[0]["token_str"]
                else:
                    predicted_token = tokens[idx]  # Fallback to original token if prediction fails

                new_tokens = tokens.copy()
                new_tokens[idx] = predicted_token

                final_sentence = tokenizer.convert_tokens_to_string(new_tokens)
                modified_adv_texts.append(final_sentence)

                if verbose:
                    print(f"[Index {idx} replaced]: {final_sentence}")

    return modified_adv_texts
