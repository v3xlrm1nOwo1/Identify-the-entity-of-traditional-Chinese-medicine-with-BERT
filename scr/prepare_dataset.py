import pandas as pd
import config
import datasets
from sklearn import preprocessing
from sklearn import model_selection



def process_data(dataset_path=config.DATASET_PATH):
    
    # Load Dataset
    ds = datasets.load_dataset(dataset_path)
    
    # Convert to pandas
    ds.set_format(type='pandas')
    df = ds['train'][:]
    
    # CMEEE: Chinese Medical CBLUE Traditional Chinese Medicine Entity Recognition Dataset
    cmeee_df = df[df['task_name'] == 'cmeee']

    # covert text and labels to list
    cmeee_df['text_list'] = cmeee_df.apply(func= (lambda df: list(str(df['text']))), axis=1)
    cmeee_df['entities'] = cmeee_df.apply(func=(lambda df: str(df.labels).split()), axis=1)

    assert (cmeee_df['text_list'].str.len() == cmeee_df['entities'].str.len()).all(), "Lengths of text_split and entities do not match for all rows."

    # Labels Encoder
    labels = cmeee_df['labels'].to_numpy()
    
    labels_str = ''
    for label in labels:
        labels_str += f' {label}'

    labels = labels_str.split()
    labels = list(set(labels))

    label_encoder = dict()
    for idx, label in enumerate(labels):
        label_encoder[label] = idx
        
    cmeee_df['labels_encoder'] = cmeee_df.apply(func=(lambda df: [label_encoder[label] for label in df.entities]), axis=1)
    
    tarin_cmeee_df, test_cmeee_df = model_selection.train_test_split(cmeee_df, random_state=config.RANDOM_SEED, test_size=0.1)
    test_cmeee_df, valid_cmeee_df = model_selection.train_test_split(test_cmeee_df, random_state=config.RANDOM_SEED, test_size=0.5)
    
    return tarin_cmeee_df, valid_cmeee_df, test_cmeee_df
