import torch
import transformers


MAX_LEN = 260
NUM_EPOCHS = 30
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
DATASET_PATH = 'ttxy/cn_ner'
LABELS_ENTITIES = {'I-sym': 0, 'B-equ': 1, 'B-mic': 2, 'I-ite': 3, 'B-dep': 4, 'O': 5, 'B-ite': 6, 'B-pro': 7, 'I-dru': 8, 'I-mic': 9, 'I-dep': 10, 'I-dis': 11, 'B-sym': 12, 'I-pro': 13, 'I-bod': 14, 'B-bod': 15, 'I-equ': 16, 'B-dru': 17, 'B-dis': 18}
NUM_ENTITY = len(LABELS_ENTITIES.keys())

LEARN_RATE = 3e-6
CHECKPOINT = 'bert-base-chinese'
TOKENIZER = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 666

MODEL_PATH = '/content/drive/MyDrive/Machine_Learning_Models/xyz/best_model.pt'
