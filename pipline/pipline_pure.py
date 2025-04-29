
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f'current work patth {current_dir}')
sys.path.append(current_dir)

import time
import json
import torch
import numpy as np
import logging
import random
from PIL import Image
from torchvision import transforms
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, pipeline
from ruamel.yaml import YAML
from Defence.Text.models.blip_vqa import blip_vqa
from Defence.Text.eval_utils import MaskDemaskWrapper, MaskDemaskWrapperLLaVA
from pipline_utils import bit_depth_reduction, median_filtering, l2_distance
from scipy.stats import entropy
from scipy.ndimage import median_filter
from transformers import BertTokenizer
import torch.backends.cudnn as cudnn
import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from typing import Callable, Any, List, Tuple, Optional

#print("Current working directory:", os.getcwd())

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def load_json_data(
    file_path: str,
    transform: Callable,
    device: Any,
    end_index: int,
    mode: str = 'inference'
) -> List[Tuple[Any, str, str, Optional[str]]]:

    def process_entry(question: str, image_path: str, label: Optional[str]) -> Optional[Tuple[Any, str, str, Optional[str]]]:
        if not question or not image_path:
            logger.warning(f"Missing question or image_path for label: {label}")
            return None
        full_path = os.path.join("./static/uploads", f"upload_{image_path}")
        image_tensor = load_image(full_path, transform, device)
        if image_tensor is None:
            return None
        return (image_tensor, question, full_path, label)

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc=f"Loading JSON data ({mode})"):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue

                if mode == 'inference':
                    result = process_entry(
                        json_obj.get("question"),
                        json_obj.get("image_path"),
                        label=None
                    )
                    if result:
                        data.append(result)

                elif mode == 'test':
                    # benign sample
                    benign = process_entry(
                        json_obj.get("original_question"),
                        json_obj.get("original_image_path"),
                        label="benign"
                    )
                    if benign:
                        data.append(benign)

                    # adversarial sample
                    adv = process_entry(
                        json_obj.get("adversarial_question"),
                        json_obj.get("adversarial_image_path"),
                        label="adversarial"
                    )
                    if adv:
                        data.append(adv)

                if len(data) >= end_index:
                    break
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")

    return data


def load_image(image_path, transform, device):
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def initialize_model_and_tokenizer(config, device):
    text_tokenizer = init_tokenizer()
    vqa_model = blip_vqa(
        pretrained=config['pretrained'],
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer']
    ).to(device)
    return vqa_model, text_tokenizer

def initialize_masked_lm(bert_model_name, device):
    model = BertForMaskedLM.from_pretrained(bert_model_name).to(device)
    tokenizer = init_tokenizer()
    return pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)

def detect_adversarial_image_example_1(image, question, answer_tokens, candidate_answers, vqa_model, pro_softmax_original, threshold=0.7, VLModel_name='BLIP', processor=None, original_pred=None):
    predictions = []
    compressed_image_1 = bit_depth_reduction(image, bits=4)
    compressed_image_2 = median_filtering(image, kernel_size=3)
    if VLModel_name.lower() == 'blip':
        distances = []
        for compressed_image in [compressed_image_1, compressed_image_2]:
            max_ids, _, _, pro_softmax = vqa_model(
                compressed_image,
                question=[question],
                answer=answer_tokens,
                train=False,
                inference='rank',
                k_test=10
            )
            prediction = candidate_answers[int(max_ids.cpu().numpy()[0])]
            distance = l2_distance(pro_softmax, pro_softmax_original)
            predictions.append(prediction)
            distances.append(distance)
        if distances[0] > threshold:
            return True, prediction
        return False, prediction
    elif VLModel_name.lower() == 'llava':
        prompt = format_llava_prompt(question)
        inputs = processor(text=prompt, images=compressed_image_1, return_tensors="pt", do_rescale=False).to(vqa_model.device)
        generate_ids = vqa_model.generate(**inputs, max_new_tokens=100)
        prediction = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        if prediction != original_pred:
            return True, prediction
        return False, prediction

def detect_adversarial_image_example_2(image, question, answer_tokens, candidate_answers, vqa_model, pro_softmax_original, threshold=0.7, VLModel_name='BLIP', processor=None, original_pred=None):
    predictions = []
    compressed_image_1 = bit_depth_reduction(image, bits=4)
    compressed_image_2 = median_filtering(image, kernel_size=3)
    if VLModel_name.lower() == 'blip':
        distances = []
        for compressed_image in [compressed_image_1, compressed_image_2]:
            max_ids, _, _, pro_softmax = vqa_model(
                compressed_image,
                question=[question],
                answer=answer_tokens,
                train=False,
                inference='rank',
                k_test=10
            )
            prediction = candidate_answers[int(max_ids.cpu().numpy()[0])]
            distance = l2_distance(pro_softmax, pro_softmax_original)
            predictions.append(prediction)
            distances.append(distance.item())
            print(distances)
        if distances[1] > threshold or distances[0] > threshold:
            return True, prediction
        return False, prediction
    elif VLModel_name.lower() == 'llava':
        prompt = format_llava_prompt(question)
        inputs = processor(text=prompt, images=compressed_image_2, return_tensors="pt", do_rescale=False).to(vqa_model.device)
        generate_ids = vqa_model.generate(**inputs, max_new_tokens=100)
        prediction = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        if prediction != original_pred:
            return True, prediction
        return False, prediction

def detect_adversarial_text_example(ag_wrapper, predict_origi, question, image):
    prediction_text = ag_wrapper([question], [image])
    if prediction_text != predict_origi:
        return True, prediction_text
    return False, prediction_text

def detect_adversarial_textandimage_example(ag_wrapper, predict_origi, question, image):
    compressed_image_1 = bit_depth_reduction(image, bits=4)
    compressed_image_2 = median_filtering(image, kernel_size=3)
    prediction_text_transform_1 = ag_wrapper([question], [compressed_image_1])
    prediction_text_transform_2 = ag_wrapper([question], [compressed_image_2])
    if prediction_text_transform_1 != predict_origi or prediction_text_transform_2 != predict_origi:
        return True
    return False

def format_llava_prompt(question: str) -> str:
    return f"USER: <image>\n{question.strip()} Answer with one word only.\nASSISTANT:"

def main(
    config_path='./pipline/configs/vqa.yaml',
    data_file='../static/uploads/upload_info.json',# web input
    answer_list_file='./pipline/Defence/Text/answer_list.json',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    bert_model_name='bert-large-uncased',
    mask_pct=0.1,
    num_voter=5,
    max_samples=4000,
    #VLmodel_name='LLaVA',
    VLmodel_name='BLIP',                 # web input
    image_detector='feature_squeezing_1',# web input
    text_detector='makepure',            # web input
    image_text_detector='jointdetection', # web input
    #mode = 'inference'          
    mode = 'test'

):
    logger.info(f"Using device: {device}")

    yaml = YAML(typ="safe")
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    seed = 42
    np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if VLmodel_name.lower() == 'blip':
        vqa_model, text_tokenizer = initialize_model_and_tokenizer(config, device)
    elif VLmodel_name.lower() == 'llava':
        model_id = "llava-hf/llava-1.5-7b-hf"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)
        vqa_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        text_tokenizer = processor.tokenizer
    logger.info(f"Model and tokenizer of {VLmodel_name} loaded successfully.")

    with open(answer_list_file, "r", encoding="utf-8") as f:
        candidate_answers = json.load(f)
    if VLmodel_name.lower() == 'blip':
        answer_tokens = text_tokenizer(candidate_answers, padding='longest', return_tensors="pt")
        answer_tokens = {k: v.to(device) for k, v in answer_tokens.items()}
    else:
        answer_tokens = None

    ag_fill_mask = initialize_masked_lm(bert_model_name, device)
    if VLmodel_name.lower() == 'blip':
        ag_wrapper = MaskDemaskWrapper(
            vqa_model, text_tokenizer, ag_fill_mask, num_voter, mask_pct,
            'rank', answer_tokens, candidate_answers
        )
    elif VLmodel_name.lower() == 'llava':
        ag_wrapper = MaskDemaskWrapperLLaVA(
            vqa_model, processor, num_voter, mask_pct, 'generate', 
            answer_tokens, candidate_answers, max_new_tokens=10, 
            fill_mask=ag_fill_mask
        )

    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
    ])
    dataset = load_json_data(data_file, transform, device, end_index=max_samples,mode=mode)
    logger.info(f"Loaded {len(dataset)} samples.")

    adversarial_example_count = 0
    total_count = 0
    result_records = []
    false_positive, true_positive, false_negative = 0, 0, 0
    
    precision, recall = 0, 0

    for idx, (image, question, fullpath, label) in enumerate(dataset):
        try:
            total_count += 1
            print('--------------------------------')
            print(f"Sample {total_count}:")
            print(f"  Question: {question}")
            print(f"label: {label}")
            start_time = time.time()
            record = {
                "question": question,
                "image_path": fullpath,
                "prediction": None,
                "processing_time": None,
            }

            if VLmodel_name.lower() == 'blip':
                max_ids, _, _, pro_softmax_original = vqa_model(
                    image,
                    question=[question],
                    answer=answer_tokens,
                    train=False,
                    inference='rank',
                    k_test=10
                )
                predict_origi = candidate_answers[int(max_ids.cpu().numpy()[0])]
            elif VLmodel_name.lower() == 'llava':
                prompt = format_llava_prompt(question)
                inputs = processor(text=prompt, images=image, return_tensors="pt", do_rescale=False).to(vqa_model.device)
                generate_ids = vqa_model.generate(**inputs, max_new_tokens=100)
                predict_origi = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

            is_adv_image = False
            if image_detector.lower() != 'none':
                if image_detector == 'feature_squeezing_1':
                    is_adv_image, preds = detect_adversarial_image_example_1(
                        image, question, answer_tokens, candidate_answers,
                        vqa_model, pro_softmax_original if VLmodel_name.lower() == 'blip' else None,
                        VLModel_name=VLmodel_name,
                        processor=processor if VLmodel_name.lower() == 'llava' else None,
                        original_pred=predict_origi
                    )
                elif image_detector == 'feature_squeezing_2':
                    is_adv_image, preds = detect_adversarial_image_example_2(
                        image, question, answer_tokens, candidate_answers,
                        vqa_model, pro_softmax_original if VLmodel_name.lower() == 'blip' else None,
                        VLModel_name=VLmodel_name,
                        processor=processor if VLmodel_name.lower() == 'llava' else None,
                        original_pred=predict_origi
                    )
                elif image_detector.lower() == 'none':
                    continue    
                else:
                    raise ValueError(f"Unknown image_detector: {image_detector}")

            if is_adv_image:
                adversarial_example_count += 1 
                true_positive += int(label == 'adversarial')
                false_positive += int(label == 'benign')
                print(f"!!! Detected adversarial IMAGE example: {question}")
                record["prediction"] = "Image-Only Adversarial"
                record["processing_time"] = round(time.time() - start_time, 4)
                result_records.append(record)
                continue

            is_adv_text = False
            if text_detector.lower() != 'none':
                if text_detector.lower() == 'maskpure':
                    is_adv_text, preds = detect_adversarial_text_example(ag_wrapper, predict_origi, question, image)
                elif text_detector.lower() == 'none':
                    continue
                else:
                    raise ValueError(f"Unknown text_detector: {text_detector}")

            if is_adv_text:
                adversarial_example_count += 1
                true_positive += int(label == 'adversarial')
                false_positive += int(label == 'benign')
                print(f"!!! Detected adversarial TEXT example: {question}")
                record['prediction'] = "Text-Only Adversarial"
                record["processing_time"] = round(time.time() - start_time, 4)
                result_records.append(record)
                continue

            is_adv_textandimage = False
            if image_text_detector.lower() != 'none':
                if image_text_detector.lower() == 'jointdetection':
                    is_adv_textandimage = detect_adversarial_textandimage_example(ag_wrapper, predict_origi, question, image)
                elif image_text_detector.lowe() == 'none':
                    continue
                else:
                    raise ValueError(f"Unknown image_text_detector: {image_text_detector}")

            if is_adv_textandimage:
                adversarial_example_count += 1
                true_positive += int(label == 'adversarial')
                false_positive += int(label == 'benign')
                print(f"!!! Detected adversarial TEXT+IMAGE example: {question}")
                record['prediction'] = "Text-Image-Joint Adversarial"
                record["processing_time"] = round(time.time() - start_time, 4)
                result_records.append(record)
                continue

            # input is benign example
            print(f"Benign example: {question}")
            false_negative += int(label=='benign')
            record['prediction'] = "Benign"
            record["processing_time"] = round(time.time() - start_time, 4)
            result_records.append(record)


        except Exception as e:
            logger.exception(f"Error processing sample {idx+1}: {e}")

    logger.info("Classification completed.")
    print(f"Adversarial example count: {adversarial_example_count}/{total_count}")

    if mode == 'test':
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive +false_negative)
    else:
        precision = None
        recall = None


    result_json = {
        'results': result_records,
        'image_method': image_detector,
        'text_method': text_detector,
        'joint_method': image_text_detector,
        'model_selected': VLmodel_name,
        'precision': precision,
        'recall': recall
    }
    os.makedirs("../static/uploads", exist_ok=True)
    with open("./static/uploads/result.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main(VLmodel_name='LLaVA')
