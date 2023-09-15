import torch
import torch.nn as nn

def loss_fn(output, label, mask, num_entity):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_entity)
    active_labels = torch.where(
        active_loss,
        label.view(-1),
        torch.tensor(lfn.ignore_index).type_as(label)
    )
    
    loss = lfn(active_logits, active_labels)
    return loss


def optimizer_parameters(model):
    prm_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_parameters = [
        {
            'params': [
                p for n, p in prm_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.001,
        },
        {
            'params': [
                p for n, p in prm_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    
    return opt_parameters

