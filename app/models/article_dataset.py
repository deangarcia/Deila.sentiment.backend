from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
data_files = {"train":"train.csv", "validate":"validate.csv" }
dataset = load_dataset("deancgarcia/Diversity", data_files=data_files)
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

class ArticleDataset(Dataset):
  def __init__(self, datatype, use_tokenizer):
    titles_df = dataset[datatype]['Title']
    text_df = dataset[datatype]['Content']
    labels_df = dataset[datatype]['Sentiment']
    temp_text = list(text_df)
    temp_labels = list(labels_df)
    #fix text issues clean the content and combine title
    #maybe not adding the data here need to do for loop through the temp_texts
    #print this stuff out 
    # also need to see how much text is actually going in 1024 might be to small bump it up
    self.texts = titles_df + temp_text[:1024]
    self.labels = []
    for labels in temp_labels:
      if labels:
        self.labels.append("pos")
      else:
        self.labels.append("neg")

    # Number of exmaples.
    self.n_examples = len(self.labels)
    

    return

  def __len__(self):    
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item],
            'label':self.labels[item]}
            
  def train(dataloader, optimizer_, scheduler_, device_):
    global model
    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.train()

    for batch in dataloader:
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader) 
    return true_labels, predictions_labels, avg_epoch_loss
