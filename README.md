# Adversarial Example Detection for VQA Models

This project implements a robust adversarial example detection pipeline for Visual Question Answering (VQA) models. It supports both BLIP and LLaVA, and integrates image, text, and joint detection strategies for identifying adversarial inputs.



## Features

- Supports **BLIP** and **LLaVA** VQA models
- Multiple detection methods:
  - **Feature Squeezing** (bit-depth reduction + median filtering)
  - **Mask-Pure** method for detecting adversarial text
  - **Joint detection** (image + text)



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/peilin1011/VLmode_adv_detection.git
cd vqa-adversarial-detector
```

### 2. Set up virtual environment and install dependencies

```bash
python3 -m venv adv_env
source adv_env/bin/activate
pip install -r requirements.txt
```

## Data Preparation
Create a folder and place all your image files there. Additionally, create a JSON file named info.json in the same folder that contains metadata for each sample.

### Inference Mode

You can run the model in inference mode in two ways:

#### Option 1: File-based input

{
  "question": "is the boy playing baseball !",
  "image_path": "131089004_adversarial.png"
}

question: the question to be answered by the model

image_path: filename of the image (must be placed in ./static/uploads/)

#### Option 2: Web Interface

You can directly upload an image and input a question via the web interface. The Flask server will handle the detection and return the result.

### Test Mode

{
  "question_id": 393284015,
  "original_question": "what color is the man's jacket?",
  "adversarial_question": "what color is the man s his jacket?",
  "original_image_path": "393284015_original.png",
  "adversarial_image_path": "393284015_adversarial.png",
  "original_answer": "red",
  "adversarial_answer": "yellow",
  "attack_type": "image_text"
}

This format is used for evaluation and benchmarking.

Each line in the JSONL should contain both benign and adversarial examples with corresponding paths and questions.

## Run the Detector

```bash
python3 app.py
```


## Project Structure

```
.
├── app.py                   # Main script
├── Defence/                 # Core detection logic
├── pipline_utils.py         # Image transformation helpers
├── static/uploads/          # Processed images
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```



## Acknowledgements

- [BLIP](https://github.com/salesforce/BLIP)
- [LLaVA](https://github.com/haotian-liu/LLaVA)



##  License

MIT License