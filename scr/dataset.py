import config
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class EntityDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: config.TOKENIZER, max_len: int = config.MAX_LEN, include_row_text=False):
        self.data = data
        self.tokenizer = tokenizer
        self.include_row_text = include_row_text
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        data_row = self.data.iloc[item]
        sentence = data_row['text_list']
        entities = data_row['labels_encoder']
        
        input_ids = []
        labels_entities = []

        for i, word in enumerate(sentence):
            inputs = self.tokenizer.encode(word, add_special_tokens=False)

            input_len = len(inputs)
            
            input_ids.extend(inputs)
            labels_entities.extend([entities[i]] * input_len)

        # cut input_ids and entities to the max length
        input_ids = input_ids[: self.max_len - 2]
        labels_entities = labels_entities[: self.max_len - 2]

        # Add special tokens
        input_ids = [101] + input_ids + [102]
        labels_entities = [0] + labels_entities + [0]

        # Add attention mask and token type ids
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Add pandding to model inputs
        padding_len = self.max_len - len(input_ids)

        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        labels_entities = labels_entities + ([0] * padding_len)

        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels_entities': torch.tensor(labels_entities, dtype=torch.long),
        }
        
        if self.include_row_text:
            output['sentences'] = data_row['text']
            
        return output
        
     

def create_data_loader(data, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.TRAIN_BATCH_SIZE, include_row_text=False, shuffle=False):
  ds = EntityDataset(
    data=data,
    tokenizer=tokenizer,
    max_len=max_len,
    include_row_text=include_row_text
  )

  return DataLoader(
    ds,
    shuffle=shuffle,
    batch_size=batch_size,
    num_workers=4,
  )   
