import numpy as np
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import prepare_dataset
import dataset
import utils
import engine
from ner_model import EntityModel

device = config.DEVICE
labels_encoder = config.LABELS_ENTITIES

tarin_cmeee_df, valid_cmeee_df, test_cmeee_df = prepare_dataset.process_data(config.DATASET_PATH)

train_data_loader = dataset.create_data_loader(tarin_cmeee_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
valid_data_loader = dataset.create_data_loader(valid_cmeee_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.VALID_BATCH_SIZE, shuffle=False) 
test_data_loader = dataset.create_data_loader(test_cmeee_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.TEST_BATCH_SIZE, include_row_text=False, shuffle=False)

model = EntityModel(num_entity=len(labels_encoder.values()))
model.to(device)



num_train_steps = int(len(train_data_loader) / config.TRAIN_BATCH_SIZE * config.NUM_EPOCHS)

optimizer_parameters = utils.optimizer_parameters(model)
optimizer = AdamW(optimizer_parameters, lr=config.LEARN_RATE)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)


best_loss = np.inf
for epoch in range(config.NUM_EPOCHS):
    train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
    
    valid_loss = engine.eval_fn(valid_data_loader, model, device)
    
    print('=' * 50)
    print(f'({epoch + 1} / {config.NUM_EPOCHS}) Train Loss = {train_loss} - Valid Loss = {valid_loss}')
    print('=' * 50)
    
    if valid_loss < best_loss:
        torch.save(model.state_dict(), config.MODEL_PATH)
        best_loss = valid_loss


test_loss = engine.test_fn(test_data_loader, model, device)

print('=' * 50)
print(f'Test Loss = {train_loss}')
print('=' * 50)