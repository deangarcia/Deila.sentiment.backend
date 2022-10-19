from datasets import load_dataset
data_files = {"train":"train.csv", "validate":"validate.csv" }
dataset = load_dataset("deancgarcia/Diversity", data_files=data_files)

import sys
sys.path.append("C:/Users/Dean/Documents/GitHub/DEILA/Deila.sentiment.backend/app/models")
from classification_collator import Gpt2ClassificationCollator
from article_dataset import ArticleDataset
import io
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification) 

def model_classify(zero, init_token):
    set_seed(123)
    epochs = 4
    batch_size = 32
    max_length = 60
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'gpt2'
    labels_ids = {'neg': 0, 'pos': 1}
    n_labels = len(labels_ids)

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, labels_encoder=labels_ids, max_sequence_len=max_length)
    train_dataset = ArticleDataset(datatype='train', use_tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    valid_dataset =  ArticleDataset(datatype='validate', use_tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)


    # Adaptive Moment Estimation Optimzer 
    #combined advantages of gradient descent with momentum and the RMSProp optimization algorithm
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    for epoch in range(epochs):
        ################################### Training ###########################################
        train_predict = []
        train_labels = []
        total_loss = 0

        model.train()

        for batch in train_dataloader:
            train_labels += batch['labels'].numpy().flatten().tolist()
            batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
            model.zero_grad()
            outputs = model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = logits.detach().cpu().numpy()
            train_predict += logits.argmax(axis=-1).flatten().tolist()

        train_loss = total_loss / len(train_dataloader) 
        ################################### Training ###########################################

        train_acc = accuracy_score(train_labels, train_predict)

        ################################### Validation ###########################################
        valid_predict = []
        valid_labels = []
        total_loss = 0

        model.eval()

        for batch in valid_dataloader:
            valid_labels += batch['labels'].numpy().flatten().tolist()

            batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

            with torch.no_grad():        
                outputs = model(**batch)

                loss, logits = outputs[:2]
                
                logits = logits.detach().cpu().numpy()

                total_loss += loss.item()
                
                predict_content = logits.argmax(axis=-1).flatten().tolist()

                valid_predict += predict_content

        val_loss = total_loss / len(valid_dataloader)

        ################################### Validation ###########################################
    model.eval()
    init_id = tokenizer.encode(init_token)
    result = init_id
    init_input = torch.tensor(init_id).unsqueeze(zero).to(device)

    with torch.no_grad():
        output = model(init_input)
        logits = output.logits[:2]
        logits = logits.detach().cpu().numpy()
        print(logits)
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        print(predict_content)

    return predict_content




name = model_classify(0, "Tallahassee Mayor Andrew Gillum electrified Democrats with his surprising victory in the Florida’s Democratic primary – but will he go on to win in the general election? Come November, voters will choose between Gillum and Trump-endorsed candidate U.S. Rep. Ron DeSantis. DeSantis, who represents Florida’s Sixth Congressional District, won his nomination by a significant margin. Both men are 39 years old, politically experienced Florida natives – perhaps the only two similarities they share. After trailing in the polls for weeks before the election, Gillum, who spent US$6.5 million in the primary, defeated three opponents who each spent more than $100 million in their campaigns. Gillum, the only candidate who was not a millionaire, received $650,000 in last-minute contributions from donors such as Tom Steyer and George Soros. He now joins Georgia’s Stacey Abrams and Maryland’s Ben Jealous – two other young African-Americans with strong chances of winning their state’s gubernatorial elections. Each won their Democratic primaries because of the strong backing from black voters. But because none of them could have won with the black vote alone, their campaigns emphasized issues voters of all races were concerned with, like health care, and education and jobs. All received significant backing in some predominantly white communities. Their victories are significant and rare because only four African-Americans have ever served as governors in our nation’s history – but winning during the general election won’t be an easy task. Gillum in particular is competing in a state that hasn’t elected a Democratic governor in 20 years. True, former President Barack Obama won Florida twice, but it was by close margins – 3.8 percent in 2008 and 0.9 percent in 2012. Then, President Trump again put Florida in the red category in 2016 by defeating Hillary Clinton by a mere 0.8 percent. However, as a professor of political science and African-American studies, I believe the unpredictable outcomes in recent national elections – as well as Florida’s tendency to swing from red to blue – should encourage Gillum. So how can Gillum win? He’ll need a large turnout among his base of minority voters and progressives. He’ll also need to expand his appeal among moderate Democrats and to seek crossover support from Republicans who are dissatisfied with President Trump. In the primary, he won only 18 of the state’s 67 counties. Some of these included cities and towns with larger minority populations, but others were rural or suburban predominantly white counties – like Clay, Escambia and Hamilton. Gillum also did well in South Florida counties like Broward, Hendry, Miami-Dade and Palm Beach. Unfortunately for Andrew Gillum, he won’t be running against Ron DeSantis alone. He’s also be running against Donald Trump. DeSantis is one of Trump’s most loyal allies. Hours after Gillum won the primary, Trump referred to him as “[ DeSantis’s] biggest dream … a failed socialist mayor.”A more troublesome dilemma for Gillum concerns Tallahassee’s problems. Three years after he entered office, in June 2017, the FBI issued a subpoena of city records. Although Gillum is reportedly not the focus of their corruption investigation, the investigation allows the DeSantis campaign to accuse him of being untrustworthy regardless of the outcome. Tallahassee also has the highest crime rate in Florida, even though crime has actually decreased since Gillum began his term in 2014.")

print(name)
