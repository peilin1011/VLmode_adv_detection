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
        self.model = model  # Classification model
        self.tokenizer = tokenizer  # Tokenizer
        self.fill_mask = fill_mask  # Tool or function used for mask filling
        self.num_voter = num_voter  # Number of answers generated per input
        self.mask_pct = mask_pct    # Masking percentage
        #self.pipeline = pipeline('text-classification', model=model, 
                                 #tokenizer=tokenizer, device=next(model.parameters()).device)
        self.answer_tokens = answer_tokens
        self.inference = inference
        self.candidate_answers = candidate_answers

    def __call__(self, text_input_list, images_list):
        # Generate perturbed (masked and filled) inputs
        filled = mask_and_demask(
            text_input_list, self.tokenizer, self.fill_mask,
            verbose=True, num_voter=self.num_voter, mask_pct=self.mask_pct,
            use_random_mask=True
        )
        # Extend images_list to match the length of filled inputs
        images_list = images_list * (len(filled) // len(images_list) + 1)
        images_list = images_list[:len(filled)]  # Truncate to match length

        all_answers = []  # To store all generated answers
        for (i, (text, image)) in enumerate(zip(filled, images_list)):
            if self.inference == 'generate':
                output = self.model(image=image, question=[text], train=False, inference='generate')
            elif self.inference == 'rank':
                max_ids, topk_ids, topk_probs, _ = self.model(
                    image=image, question=[text], answer=self.answer_tokens,
                    train=False, inference='rank', k_test=128
                )
                print(f"max_ids: {max_ids}")
                output = [self.candidate_answers[idx] for idx in max_ids.cpu().numpy()]
            all_answers.extend(output)

        # Count frequency of each answer
        answer_counts = Counter(all_answers)
        print(f"answer_counts: {answer_counts}")

        most_common_answer = answer_counts.most_common(1)[0][0]
        return most_common_answer


class MaskDemaskWrapperLLaVA(ModelWrapper):
    def __init__(self, model, processor, num_voter, mask_pct, 
                 answer_tokens=None, candidate_answers=None, max_new_tokens=10):
        self.model = model
        self.processor = processor
        self.num_voter = num_voter
        self.mask_pct = mask_pct
        self.answer_tokens = answer_tokens
        self.candidate_answers = candidate_answers
        self.max_new_tokens = max_new_tokens

    def __call__(self, text_input_list, images_list):
        # Generate perturbed variants
        filled = mask_and_demask_llava(
            text_input_list,
            num_voter=self.num_voter,
            mask_pct=self.mask_pct,
            verbose=True
        )
        
        #print(f'.......variant: {filled}-----------')
        # Align images to match the number of texts
        images_list = images_list * (len(filled) // len(images_list) + 1)
        images_list = images_list[:len(filled)]

        all_answers = []

        for text, image in zip(filled, images_list):
            prompt = format_llava_prompt(text)
            #print(f"[LLaVA Prompt] {prompt}")

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                do_rescale=False
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            if "ASSISTANT:" in decoded:
                answer_part = decoded.split("ASSISTANT:")[-1].strip()
            else:
                answer_part = decoded.strip()
            
            first_word = answer_part.split()[0]
            all_answers.append(first_word)

        # Majority vote
        answer_counts = Counter(all_answers)
        most_common = answer_counts.most_common(1)[0][0]
        print(f"[Answer Counts] {answer_counts}")

        return most_common
