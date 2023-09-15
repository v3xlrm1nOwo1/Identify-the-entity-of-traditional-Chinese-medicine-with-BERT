import config
import transformers
import torch.nn as nn
import utils

class EntityModel(nn.Module):
    def __init__(self, num_entity):
        super(EntityModel, self).__init__()
        self.num_entity = num_entity
        self.bert = transformers.AutoModel.from_pretrained(config.CHECKPOINT, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.linear_layer = nn.Linear(self.bert.config.hidden_size, self.num_entity)
    
    def forward(self, input_ids, attention_mask, token_type_ids, label_entity=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        output = self.drop(output)

        output = self.linear_layer(output)
        
        if label_entity != None:
            loss = utils.loss_fn(output, label_entity, attention_mask, self.num_entity)
            return output, loss
        
        else:
            return output
