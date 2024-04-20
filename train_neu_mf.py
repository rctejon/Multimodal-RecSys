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
from tensorboardX import SummaryWriter
from metrics.metrics import metrics
from architectures.NeuMF.neu_mf import NeuMF
from loaders.create_dataloader import CreateDataloader
from tqdm import tqdm
import pickle

def _reindex(ratings):
    """
    Process dataset to reindex userID and itemID, also set rating as binary feedback
    """
    user2id = pickle.load(open(MAIN_PATH + 'user2id.pkl', 'rb'))

    item2id = pickle.load(open(MAIN_PATH + 'item2id.pkl', 'rb'))

    ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
    ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
    ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
    return ratings

if __name__ == '__main__':
    DATASET_NAME = 'MOOCCubeX'
    MODEL_NAME = 'NeuMF'
    TRAIN_DATASET_FILE = 'train.feather'
    TEST_DATASET_FILE = 'test.feather'
    MAIN_PATH = f'./data/{DATASET_NAME}/'
    TRAIN_DATA_PATH = MAIN_PATH + TRAIN_DATASET_FILE
    TEST_DATA_PATH = MAIN_PATH + TEST_DATASET_FILE
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
        default=0,
        help="Number of negative samples for test set")
    parser.add_argument("--out",
        default=True,
        help="save model or not")
    
    # set device and parameters
    args = parser.parse_args()
    print(args.epochs, args.lr, args.dropout, args.batch_size, args.factor_num, args.layers, args.num_ng, args.num_ng_test, args.top_k, args.out)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    writer = SummaryWriter()

    # seed for Reproducibility
    seed_everything(args.seed)

    # load data
    print(TRAIN_DATA_PATH)
    train_rating_data = pd.read_feather(TRAIN_DATA_PATH)
    test_rating_data = pd.read_feather(TEST_DATA_PATH)
    print(train_rating_data.head())

    train_rating_data = train_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})
    test_rating_data = test_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})

    ratings = pd.concat([train_rating_data, test_rating_data], ignore_index=True)

    # set the num_users, items
    num_users = ratings['user_id'].nunique()+1
    num_items = ratings['item_id'].nunique()+1

    print(num_users, num_items)

    train_rating_data = _reindex(train_rating_data)
    test_rating_data = _reindex(test_rating_data)


    # construct the train and test datasets

    data = CreateDataloader(args, train_rating_data, test_rating_data)
    print('Create Train Data Loader')
    train_loader = data.get_train_instance()

    # set model and loss, optimizer
    model = NeuMF(args, num_users, num_items)
    model = model.to(device)
    print(model)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # train, evaluation
    best_hr = 0
    for epoch in range(1, args.epochs+1):
        model.train() # Enable dropout (if have).
        start_time = time.time()

        for user, item, label in tqdm(train_loader):
            # print(user.size(), item.size(), label.size())
            # print(user, item, label)
            
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # print('Zero Grad')
            prediction = model(user, item)
            # print('Prediction')
            loss = loss_function(prediction, label)
            # print('Loss')
            loss.backward()
            # print('Backward')
            optimizer.step()
            # print('Step')
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
        print('epoch time: {:.4f}s'.format(time.time()-start_time))

        model.eval()

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model,
        '{}{}.pth'.format(MODEL_PATH, MODEL))

    print('Train done')