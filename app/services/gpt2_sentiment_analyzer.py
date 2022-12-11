import sys

import io
import os
import torch
import torchvision.models as models
from transformers import (GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
labels_ids = {'neg': 0, 'pos': 1}
n_labels = len(labels_ids)

model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
# Get model's tokenizer.
#print('Loading tokenizer...')
# load the saved model here
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="/home/deila_weights.pth", config=model_config)
# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id
model.to(device)

## Only do the static app and as the next step want to integrate the local app. 
def model_classify(zero, init_token):
    #let it train here show it in demo
    global tokenizer
    global model
    model.eval()
    init_id = tokenizer.encode(init_token)
    result = init_id
    init_input = torch.tensor(init_id).unsqueeze(zero).to(device)

    with torch.no_grad():
        output = model(init_input)
        logits = output.logits[:2]
        logits = logits.detach().cpu().numpy()
        #print(logits)
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        #print(predict_content)
    return logits

print("Model Loaded")
