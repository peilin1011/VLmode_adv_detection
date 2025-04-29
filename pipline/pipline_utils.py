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
from scipy.stats import entropy
from scipy.ndimage import median_filter
from transformers import BertTokenizer
import torch.backends.cudnn as cudnn
import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from typing import Callable, Any, List, Tuple, Optional

def bit_depth_reduction(image, bits=5, device='cuda:0'):
    image = image.cpu().numpy()
    image = np.round(image * (2 ** bits - 1)) / (2 ** bits - 1)
    return torch.tensor(image).to(device)

def median_filtering(image, kernel_size=2, device='cuda:0'):
    image = image.cpu().numpy()
    filtered_image = median_filter(image, size=(1, 1, kernel_size, kernel_size))
    return torch.tensor(filtered_image).to(device)

def l1_distance(x1, x2):
    return torch.sum(torch.abs(x1 - x2))

def l2_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2))

def kl_divergence(x1, x2):
    x1 = x1.cpu().detach().numpy()
    x2 = x2.cpu().detach().numpy()
    return entropy(x1.T, x2.T)
