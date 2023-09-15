



## Identify the Entity of Traditional Chinese Medicine with BERT Chinese


### Checkpoint

BERT Base Chinese
This model has been pre-trained for Chinese, training and random input masking has been applied independently to word pieces (as in the original BERT paper).
<a href='https://arxiv.org/abs/1810.04805'>Paper</a>

### Dataset

The Dataset is <a href='https://huggingface.co/datasets/ttxy/cn_ner'>cn ner
</a> From  <a href='https://huggingface.co/ttxy'>ttxy</a>
From cn_ner dataset just I use <a href='http://www.cips-chip.org.cn/2021/CBLUE'>CMEEE dataset</a>, CMEEE dataset is Chinese Medical Information Processing Challenge List CBLUE Traditional Chinese Medicine Entity Recognition Dataset.



### Finetuning

In this project I used PyTorch and the model and dataset in the top and in the deep in traning loop I used 30 epochs and 3e-6 for learning rate, you can fined more about  finetuning step in the model file (config.py, ner_model.py and engine.py etc..)

### Run This Project
For run this project follow this steps:
- Optinal open config.py file edit MODEL_PATH to the file you want save the model.
-  Run train.py file.
```zsh
python train.py
```
- After the train model run the get_prdict.py file for get prediction of Traditional Chinese Medicine Phrases.
```zsh
python get_prdict.py
```

### Note

I did not have the resources, such as the Internet, electricity, device, etc., to train the model well and choose the appropriate learning rate, so there were no results.


> To contribute to the project, please contribute directly. I am happy to do so, and if you have any comments, advice, job opportunities, or want me to contribute to a project, please contact me <a href='mailto:V3xlrm1nOwo1@gmail.com' target='blank'>V3xlrm1nOwo1@gmail.com</a>

