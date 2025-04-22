from textattack.models.wrappers import ModelWrapper
from transformers import pipeline
import sys
from torchvision import transforms  
from PIL import Image
import os
sys.path.append('../')
from Defence.Text.utils import *
sys.path.pop()

# prompt_utils.py
def format_llava_prompt(question: str) -> str:
    return f"USER: <image>\n{question.strip()} Answer with one word only.\nASSISTANT:"



from collections import Counter
class MaskDemaskWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, fill_mask, num_voter, mask_pct, inference, answer_tokens=None, candidate_answers=None):
        self.model = model # 分类模型
        self.tokenizer = tokenizer # 分词器
        self.fill_mask = fill_mask  # 用于掩码填充的工具或函数
        self.num_voter = num_voter  # 每个输入生成的答案数量
        self.mask_pct = mask_pct    # 掩码比例
        #self.pipeline = pipeline('text-classification', model=model, 
                                 #tokenizer=tokenizer, device=next(model.parameters()).device)
        self.answer_tokens = answer_tokens
        self.inference = inference
        self.candidate_answers = candidate_answers

    def __call__(self, text_input_list, images_list):
        #print(f"Processing {len(text_input_list)} inputs...")
        #print(f"Processing {len(images_list)} images...")
        filled = mask_and_demask(text_input_list, self.tokenizer, self.fill_mask, verbose = True, 
                                 num_voter=self.num_voter, mask_pct=self.mask_pct, use_random_mask=True
                                )
        # 将 images_list 扩展到与 filled 长度一致
        images_list = images_list * (len(filled) // len(images_list) + 1)
        images_list = images_list[:len(filled)]  # 截断到与 filled 一致长度
        #print(f"Processing {len(filled)} filled inputs...")
        #print(f"Processing {len(images_list)} images...")

        all_answers = []  # 用于存储所有生成的答案
        for (i,(text, image)) in enumerate(zip(filled, images_list)):
        #for (i,(text, image)) in enumerate(zip(text_input_list, images_list)):
            if self.inference == 'generate':
                output = self.model(image=image, question=[text], train=False, inference='generate')
            elif self.inference == 'rank':
                 max_ids, topk_ids, topk_probs, _ = self.model(image=image, question=[text], answer=self.answer_tokens, train=False, inference='rank', k_test=128)
                 print(f"max_ids: {max_ids}")

                 output = [self.candidate_answers[idx] for idx in max_ids.cpu().numpy()]
            all_answers.extend(output)

            # 统计答案出现的频率
        answer_counts = Counter(all_answers)
        print(f"answer_counts: {answer_counts}")

        most_common_answer = answer_counts.most_common(1)[0][0]
        return most_common_answer
    

from collections import Counter
import torch

class MaskDemaskWrapperLLaVA(ModelWrapper):
    def __init__(self, model, processor, num_voter, mask_pct, inference, 
                 answer_tokens=None, candidate_answers=None, max_new_tokens=10, fill_mask=None):
        self.model = model  # LLaVA 模型（LlavaForConditionalGeneration）
        self.processor = processor  # AutoProcessor
        self.num_voter = num_voter
        self.mask_pct = mask_pct
        self.inference = inference  # 'generate' or 'rank'
        self.answer_tokens = answer_tokens
        self.candidate_answers = candidate_answers
        self.max_new_tokens = max_new_tokens
        self.fill_mask = fill_mask


    def __call__(self, text_input_list, images_list):

        filled = mask_and_demask(
            text_input_list,
            tokenizer=self.processor.tokenizer,
            fill_mask=self.fill_mask,
            verbose=True,
            num_voter=self.num_voter,
            mask_pct=self.mask_pct,
            use_random_mask=True
        )

        images_list = images_list * (len(filled) // len(images_list) + 1)
        images_list = images_list[:len(filled)]

        all_answers = []

        for text, image in zip(filled, images_list):
            prompt = format_llava_prompt(text)
            inputs = self.processor(text=prompt, images=image, return_tensors="pt",do_rescale = False).to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            one_word = decoded.strip().split()[0]  # 取第一个词（保守处理）
            all_answers.append(one_word)

        answer_counts = Counter(all_answers)
        print(f"answer_counts: {answer_counts}")

        most_common_answer = answer_counts.most_common(1)[0][0]
        return most_common_answer

