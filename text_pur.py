import os
import json
import torch
import numpy as np
import logging
from PIL import Image
from torchvision import transforms
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, pipeline
from ruamel.yaml import YAML
from models.blip_vqa import blip_vqa
from models.blip import init_tokenizer
from eval_utils import MaskDemaskWrapper
import os

# 替换下面的路径为你的目标路径
path = '/app'
os.chdir(path)

# 打印当前工作目录来确认更改
print("当前工作目录:", os.getcwd())

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 数据加载
def load_json_data(file_path, transform, device):
    """加载 JSON 数据集并进行预处理"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                text = json_obj["adversarial_question"]
                label = json_obj["original_answer"]
                image_path = os.path.join('./', json_obj["original_image_path"])
                image_tensor = load_image(image_path, transform, device)
                if image_tensor is not None:
                    data.append((image_tensor, text, label))
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
    return data

def load_image(image_path, transform, device):
    """加载并预处理图像"""
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

# 初始化模型和分词器
def initialize_model_and_tokenizer(config, device):
    """加载模型和分词器"""
    ag_tokenizer = init_tokenizer()
    ag_model = blip_vqa(
        pretrained=config['pretrained'],
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer']
    ).to(device)
    return ag_model, ag_tokenizer

# 初始化 Masked LM
def initialize_masked_lm(model_name, device):
    """加载 Masked Language Model"""
    model = BertForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = init_tokenizer()
    return pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)

# 主程序
if __name__ == '__main__':
    # 参数设置
    args = {
        'config': './configs/vqa.yaml',
        'data_file': './Defence/Text/adversarial_example_text.json',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'model_name': 'bert-large-uncased',
        'mask_pct': 0.9,
        'num_voter': 1
    }
    logger.info(f"Using device: {args['device']}")

    # 加载配置
    yaml = YAML(typ="safe")
    with open(args['config'], 'r') as f:
        config = yaml.load(f)

    # 设置随机种子
    np.random.seed(42)
    torch.cuda.empty_cache()

    # 初始化模型、分词器、Masked LM
    ag_model, ag_tokenizer = initialize_model_and_tokenizer(config, args['device'])
    ag_fill_mask = initialize_masked_lm(args['model_name'], args['device'])
    logger.info("Model and tokenizer loaded successfully.")

    # 数据加载与预处理
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = load_json_data(args['data_file'], transform, args['device'])
    dataset = dataset[:10]  # 仅处理前 10 个样本
    logger.info(f"Loaded {len(dataset)} samples.")

    # 加载候选答案
    with open("Defence/Text/answer_list.json", "r", encoding="utf-8") as f:
        candidate_answers = json.load(f)
    answer_tokens = ag_tokenizer(candidate_answers, padding='longest', return_tensors="pt")
    answer_tokens = {k: v.to(args['device']) for k, v in answer_tokens.items()}

    # 初始化 MaskDemaskWrapper
    ag_wrapper = MaskDemaskWrapper(
        ag_model, ag_tokenizer, ag_fill_mask, args['num_voter'], args['mask_pct'],
        'rank', answer_tokens, candidate_answers
    )

    # 推理
    for idx, (image, text, label) in enumerate(dataset):
        try:
            prediction = ag_wrapper([text], [image])
            # 打印结果
            print(f"Sample {idx+1}:")
            print(f"  Text: {text}")
            print(f"  Prediction: {prediction}")
            print(f"  Original Label: {label}")
        except Exception as e:
            logger.error(f"Error processing sample {idx+1}: {e}")

    logger.info("Classification completed.")
