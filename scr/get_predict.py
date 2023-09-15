import config
from ner_model import EntityModel
import torch
import dataset
import pandas as pd
from tqdm import tqdm


def get_predict():
    # Get sentence
    sentence  = input('Enter Text: ')
    
    # convert sentence to tokenizer
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    
    # Add fake labels 
    sentence_split  = list(sentence)
    labels_encoder = [0] * len(sentence_split)
    sentence_tokenized = config.TOKENIZER.encode(sentence)

    # convert input to data frame
    sentence_df = pd.DataFrame(columns=['text_list', 'labels_encoder'])
    sentence_df.loc[0] = [sentence_split, labels_encoder]

    # create data loder
    sentence_data_loader = dataset.create_data_loader(data=sentence_df)

    # model 
    device = config.DEVICE
    model = EntityModel(num_entity=config.NUM_ENTITY)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        for data in tqdm(sentence_data_loader, total=len(sentence_data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
                    
            output = model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                token_type_ids=data['token_type_ids'],
                label_entity=None
            )
            
            # Convert ouput
            entities = output.argmax(2).cpu().numpy().reshape(-1)[: len(sentence_tokenized)]  
            entities = entities[1: -1]
            
            sentence_tokenized = config.TOKENIZER.decode(sentence_tokenized).split()[1: -1]

            labels_entities = config.LABELS_ENTITIES

            result_dict = dict()
            
            for idx, word in enumerate(sentence_tokenized):
                for k, v in labels_entities.items():
                    if entities[idx] == v:   
                        result_dict[word] = [k, entities[idx]]
                
            print(f'sentence: {sentence}')
            print(f'sentence_tokenized: {sentence_tokenized}')
            print(f'entities: {entities}')
            print(f'Result: {result_dict}')


if __name__ == '__main__':
    get_predict()
