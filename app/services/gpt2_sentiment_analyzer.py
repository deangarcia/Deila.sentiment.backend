import sys
import io
import os
import torch
import torchvision.models as models
from transformers import (GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification)

## Only do the static app and as the next step want to integrate the local app. 
def model_classify(zero, init_token):
    return [0,1]
