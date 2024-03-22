import os
import json
import random
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional, Tuple, Union
from functools import partial
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, BertForSequenceClassification

from torch import distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from PIL import Image

class BertDataset(Dataset):
    def __init__(self,
                 data_path):
        super().__init__()
        with open(data_path) as f:
            datas = f.readlines()
        self.datas = datas

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        data = data.replace('\t', '').replace('\n', '')
        text = data[:-1]
        label = int(data[-1])
        return text, label
        
def collate_fn(batch, tokenizer):

    texts = []
    labels = []

    for text, label in batch:
        texts.append(text)
        labels.append(label)
    inputs = tokenizer(texts, return_pt=True, padding=True)
    inputs['labels'] = torch.LongTensor(labels)
    
    return inputs

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data", help='root path of dataset')
    parser.add_argument('--model_path', type=str, default="ckpts/bert-base-chinese", help='root path of model')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--ckpt', type=str, default='', help='model pretrained ckpt')
    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=1, help='print loss iterations')
    # parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    # parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default=".", help='root path')
    parser.add_argument('--work_dir', type=str, default="checkpoints", help='work directory')
    parser.add_argument('--save_dir', type=str, default="bert", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    # parser.add_argument('--save_iters', type=int, default=4000, help='save iterations')

    args = parser.parse_args()
    return args

def custom_mse(pred, target):
    return ((target-pred)**2).sum(-1).mean()
def get_optimizer(args, model):
    if args.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)
    
if __name__ == "__main__":
    args = parse_option()
    torch.cuda.set_device(args.local_rank)
    
    accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device
    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
    
    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path)
    model.classifier = nn.Linear(768, 6)
    model.config.num_labels = 6
    model.num_labels = 6
    
    train_dataset = BertDataset(os.path.join(args.dataset_path, 'train.txt'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    model.to(device=device)
    # model.half()
    optimizer = get_optimizer(args, model.model)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * args.epochs) // args.gradient_accumulation_steps,
    )
    # model, clip_text_encoder, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, clip_text_encoder, optimizer, lr_scheduler, train_loader)
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader)
    total_iters = 0
    os.makedirs(os.path.join(args.root_path, args.work_dir), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        # new epoch
        # if args.local_rank == 0:
        accelerator.print("------start epoch {}------".format(epoch))
        # train_sampler.set_epoch(epoch)
        # training
        model.train()
        
        for batch_idx, inputs in enumerate(train_loader):
            total_iters += 1
            samples = inputs['labels'].shape[0]
            for key, value in inputs.items():
                if type(value) == torch.Tensor:
                    inputs[key] = value.to(device=args.local_rank)
            outputs = model(**inputs)
            loss = outputs.loss
            # loss = loss_fn(pred_tokens.reshape[:, index, :](-1, dim), target_tokens[:, index, :].reshape(-1, dim))
            accelerator.backward(loss)
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                    epoch, batch_idx // args.gradient_accumulation_steps, len(train_loader) // args.gradient_accumulation_steps,
                        100. * batch_idx / len(train_loader), loss.item()))
            
            # save model
            if total_iters % (args.save_iters * args.gradient_accumulation_steps) == 0 and total_iters != 0:
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + '.ckpt')
                accelerator.print("save model to {}".format(save_path))
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

    model.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
    tokenizer.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
    
    