import torch

class Gpt2ClassificationCollator(object):
   def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=60):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

        self.labels_encoder = labels_encoder

        return
   def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]

        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)

        inputs.update({'labels':torch.tensor(labels)})

        return inputs