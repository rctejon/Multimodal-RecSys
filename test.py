import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import re
from tensorboardX import SummaryWriter
from metrics.metrics import metrics
from architectures.BertMF.bert_mf import BertMF
from loaders.create_dataloader import CreateDataloader

DATASET_NAME = 'ml-1m'
MODEL_NAME = 'BertMF'
DATASET_FILE = 'ratings.dat'
TEXT_INFO_FILE = 'movies.dat'
MAIN_PATH = f'./data/{DATASET_NAME}/'
DATA_PATH = MAIN_PATH + DATASET_FILE
TEXT_INFO_PATH = MAIN_PATH + TEXT_INFO_FILE
MODEL_PATH = f'./models/{DATASET_NAME}/'
MODEL = f'{DATASET_NAME}-{MODEL_NAME}'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--seed",
	type=int,
	default=42,
	help="Seed")
parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.2,
	help="dropout rate")
parser.add_argument("--batch_size",
	type=int,
	default=256,
	help="batch size for training")
parser.add_argument("--epochs",
	type=int,
	default=10,
	help="training epoches")
parser.add_argument("--top_k",
	type=int,
	default=10,
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
    nargs='+',
    default=[64,32,16,8],
    help="MLP layers. Note that the first layer is the concatenation of user \
    and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument("--num_ng",
	type=int,
	default=4,
	help="Number of negative samples for training set")
parser.add_argument("--num_ng_test",
	type=int,
	default=100,
	help="Number of negative samples for test set")
parser.add_argument("--out",
	default=True,
	help="save model or not")

text_info = pd.read_csv(
	TEXT_INFO_PATH,
	sep="::",
	names = ['item_id', 'name', 'genres'],
	engine='python',
    encoding='latin-1')

text_info['name'] = text_info['name'].apply(lambda x: 'Name: ' + x)
text_info['genres'] = text_info['genres'].apply(lambda x: 'Genres: ' + re.sub(r'\|', ', ', x))
text_info['tokenization'] = (text_info['name'] + ' ' + text_info['genres']).astype("string")

text_info = text_info.drop('name', axis=1)
text_info = text_info.drop('genres', axis=1)
text_info

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './transformers/DistilBERT/tokenizer/')

text_info['tokenization'] = text_info['tokenization'].apply(lambda x: tokenizer(x, return_tensors='pt').input_ids)

model = torch.hub.load('huggingface/pytorch-transformers', 'model', './transformers/DistilBERT/model/')

for param in model.parameters():
    param.requires_grad = False

text_info

# set device and parameters
args = parser.parse_args("")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

# seed for Reproducibility
seed_everything(args.seed)

# load data
ml_1m = pd.read_csv(
	DATA_PATH,
	sep="::",
	names = ['user_id', 'item_id', 'rating', 'timestamp'],
	engine='python')

ml_1m = ml_1m.merge(text_info, how='left', on='item_id')

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

ml_1m.head()

def _reindex(ratings):
    """
    Process dataset to reindex userID and itemID, also set rating as binary feedback
    """
    user_list = list(ratings['user_id'].drop_duplicates())
    user2id = {w: i for i, w in enumerate(user_list)}

    item_list = list(ratings['item_id'].drop_duplicates())
    item2id = {w: i for i, w in enumerate(item_list)}

    ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
    ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
    ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
    return ratings

def _leave_one_out(ratings):
    """
    leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
    """
    ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
    test = ratings.loc[ratings['rank_latest'] == 1]
    train = ratings.loc[ratings['rank_latest'] == 2]
    assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
    return train[['user_id', 'item_id', 'tokenization', 'rating']], test[['user_id', 'item_id', 'tokenization','rating']]

ml_1m = _reindex(ml_1m)
train_ml_1m, test_ml_1m = _leave_one_out(ml_1m)
train_ml_1m.head()

tokenization = ml_1m[['item_id', 'tokenization']].drop_duplicates().reset_index(drop=True)
max_tensor = max(tokenization['tokenization'].apply(lambda x: x.shape[1]))
tokenization['tokenization'] = tokenization['tokenization'].apply(lambda x: torch.cat([x, torch.zeros((1, max_tensor-x.shape[1]), dtype=torch.int64)], dim=1))
tokenization

# construct the train and test datasets
data = CreateDataloader(args, train_ml_1m, test_ml_1m, True)
train_loader = data.get_train_instance()
test_loader = data.get_test_instance()

# set model and loss, optimizer
model = BertMF(args, num_users, num_items)
model = model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

a = next(iter(train_loader))   
print(a)