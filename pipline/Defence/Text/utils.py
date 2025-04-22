import nltk
import numpy as np
import random
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch
import datasets
import pandas as pd
import random
from transformers import pipeline, BertTokenizer, BertForMaskedLM

""" def mask_and_demask(
    filtered_dataset_text,
    tokenizer,
    fill_mask,
    num_voter=5,   
    verbose=False,  
    mask_pct=0.2,  
    pos_weights=None
):
    
    # 对输入的每个句子，逐个替换每个 token，并为每个替换生成一个新的句子。
    

    modified_adv_texts = []  # 存储所有生成的新句子

    for i, example in enumerate(filtered_dataset_text):

        # 1) 用 nltk 分词 (可换成其它分词方式)
        tokens = word_tokenize(example)  # 将句子分词
        if verbose:
            print(f"Original Sentence {i}: {example}")

        # 2) 逐词替换，每个 token 替换生成一个句子
        for idx in range(len(tokens)):
            # 把第 idx 个 token 替换成 [MASK]
            temp_tokens = tokens.copy()
            temp_tokens[idx] = "[MASK]"

            # 用 fill_mask 来预测第 idx 个位置的单词
            text_str = " ".join(temp_tokens)
            results = fill_mask(text_str, top_k=1)  # 只拿最优预测

            # 获取最优预测的 token
            predicted_token = results[0]["token_str"]

            # 替换生成新的句子
            new_tokens = tokens.copy()
            new_tokens[idx] = predicted_token  # 替换第 idx 个 token

            # 将新句子转换为字符串
            final_sentence = tokenizer.convert_tokens_to_string(new_tokens)

            # 保存生成的新句子
            modified_adv_texts.append(final_sentence)

            # 打印生成的新句子（可选）
            if verbose:
                print(f"New Sentence (Token {idx} replaced): {final_sentence}")

    return modified_adv_texts 
"""


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
    对输入文本执行掩码和反掩码操作，支持“逐词替换”或“随机掩码”两种模式。
    
    参数：
    - filtered_dataset_text: 输入文本列表。
    - tokenizer: 分词器对象。
    - fill_mask: 用于填充掩码的模型（如 BERT 的 fill-mask pipeline）。
    - num_voter: 每条输入随机掩码生成的变体数量。
    - verbose: 是否打印生成过程的详细信息。
    - mask_pct: 随机掩码模式下，掩码的单词比例。
    - pos_weights: （可选）用于调整随机掩码时词性权重。如{'NN':2.0}让名词更易被掩码。
    - use_random_mask: 是否使用随机掩码（True 时走随机掩码逻辑，否则逐词替换）。

    返回：
    - 生成的新句子列表 (list[str])。
    """

    modified_adv_texts = []  # 存储所有生成的新句子

    for i, example in enumerate(filtered_dataset_text):
        
        # 1) 对文本进行分词（使用 nltk），得到 tokens
        tokens = word_tokenize(example)
        # replace the last token with '?'
        tokens[-1] = '?'
        

        # =========================
        # 随机掩码模式
        # =========================
        if use_random_mask:

            # a) 计算词性 (POS) 标签；如果 pos_weights 不为空，就据此生成掩码抽样概率
            pos_tags = nltk.pos_tag(tokens)  # [(word, pos), (word, pos), ...]
            weights = np.ones(len(tokens), dtype=float)  # 初始权重全为 1
            if pos_weights is not None:
                for idx, (_, pos) in enumerate(pos_tags):
                    if pos in pos_weights:
                        weights[idx] = pos_weights[pos]
            # 归一化，使所有 token 的权重和 = 1
            weights /= weights.sum()  

            # b) 复制 num_voter 份 token，形状 (num_voter, len(tokens))
            tokens_array = np.tile(tokens, (num_voter, 1))


            # c) 对每一份进行随机掩码
            #    计算要掩码的 token 数量
            num_tokens_to_mask = max(2, int(len(tokens) * mask_pct))
            #print(f"num_tokens_to_mask: {num_tokens_to_mask}")
            #    对每一行（即每份拷贝）随机选 num_tokens_to_mask 个位置
            mask_indices_list = [
                np.random.choice(
                    range(len(tokens)), 
                    size=num_tokens_to_mask, 
                    replace=False, 
                    p=weights
                )
                for _ in range(num_voter)
            ]

            # d) 在 tokens_array 中把对应位置改成 [MASK]
            tokens_array = tokens_array.astype(object)
            tokens_array[np.arange(num_voter)[:, None], mask_indices_list] = "[MASK]"

            # e) 将每一行 tokens 拼成字符串，形成 masked_sentences 列表
            v_convert_tokens_to_string = np.vectorize(tokenizer.convert_tokens_to_string, signature='(n)->()', otypes = [object])
            masked_sentences = v_convert_tokens_to_string(tokens_array).reshape(-1,1)

            #print(masked_sentences)
            #print(tokens_array)


            replace_idxs = np.argwhere(tokens_array == '[MASK]')
            # ummask the texts and save the results
            unmasked_text_suggestions = fill_mask([list(masked_sentence) for masked_sentence in masked_sentences], top_k=1)
            replacement_tokens = [token_info[0]['token_str']  for sentence in unmasked_text_suggestions for token_info in sentence]
            tokens_array[replace_idxs[:, 0], replace_idxs[:, 1]] = replacement_tokens
            unmasked = v_convert_tokens_to_string(tokens_array).reshape(-1,)
            [modified_adv_texts.append(unmasked[i]) for i in range(num_voter)]
            if verbose:
                print(modified_adv_texts)
        # =========================
        # 逐词替换模式
        # =========================
        else:
            # 逐词替换模式：对句子中每个 token 替换成 [MASK]，然后用 fill_mask 替换回一个 token
            for idx in range(len(tokens)):
                temp_tokens = tokens.copy()
                temp_tokens[idx] = "[MASK]"

                text_str = " ".join(temp_tokens)
                results = fill_mask(text_str, top_k=1)  # 只拿最优预测
                if len(results) > 0:
                    predicted_token = results[0]["token_str"]
                else:
                    predicted_token = tokens[idx]  # 如果没预测结果，就回退原 token

                new_tokens = tokens.copy()
                new_tokens[idx] = predicted_token

                final_sentence = tokenizer.convert_tokens_to_string(new_tokens)
                modified_adv_texts.append(final_sentence)

                if verbose:
                    print(f"[Index {idx} replaced]: {final_sentence}")

    return modified_adv_texts





