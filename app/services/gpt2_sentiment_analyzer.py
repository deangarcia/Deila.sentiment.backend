import sys

import io
import os
import torch
import torchvision.models as models
from transformers import (GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_name_or_path = 'gpt2'
#labels_ids = {'neg': 0, 'pos': 1}
#n_labels = len(labels_ids)

#model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

#tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

#tokenizer.padding_side = "left"
#tokenizer.pad_token = tokenizer.eos_token
#model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="/home/deila_weights.pth", config=model_config)
#model.resize_token_embeddings(len(tokenizer))
#model.config.pad_token_id = model.config.eos_token_id
#model.to(device)

def model_classify(zero, init_token):
    #global tokenizer
    #global model
    #model.eval()
    #init_id = tokenizer.encode(init_token)
    #result = init_id
    #init_input = torch.tensor(init_id).unsqueeze(zero).to(device)

    #with torch.no_grad():
        #output = model(init_input)
        #logits = output.logits[:2]
        #logits = logits.detach().cpu().numpy()

        #predict_content = logits.argmax(axis=-1).flatten().tolist()
    return {0,1} #logits

print("Model Loaded")
