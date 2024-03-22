from tqdm import tqdm

import torch
from torch import nn
from transformers import AutoTokenizer, BertForSequenceClassification

model_path = 'checkpoints/bert'
model_max_length = 256
# with open('data/class.txt') as f:
#     classes = f.readlines()
# label2class = {}
# for class_ in classes:
#     class_ = class_.replace('\n', '').split('   ')
#     label2class[int(class_[1])] = class_[0]

model = BertForSequenceClassification.from_pretrained(
        model_path,
    ).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                        model_max_length=model_max_length)

with open('data/test.txt') as f:
    tests = f.readlines()

predicts = []
for test in tqdm(tests):
    test = test.replace('\n', '')
    inputs = tokenizer(test, return_tensors='pt', padding=True)
    for key, value in inputs.items():
        inputs[key] = value.cuda()
    outputs = model(**inputs)
    logits = outputs.logits
    label = torch.argmax(logits, dim=-1)
    predicts.append(test + '\t' + str(label.item()) + '\n')
    
with open('data/predict.txt', 'w') as f:
    for predict in predicts:
        f.write(predict)
    