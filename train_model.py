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
from architectures.MultiMF.multi_mf import MultiMF
from architectures.BertMF.bert_mf import BertMF
from architectures.GraphMF.graph_mf import GraphMF
from architectures.NeuMF.neu_mf import NeuMF
from loaders.create_dataloader import CreateDataloader
from tqdm import tqdm
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def create_dataloader(model, main_path):
    if model == 'MultiMF':
        graph_embeddings = np.load(main_path + '/glee_embeddings.npy')
        data = CreateDataloader(args, None, None, MAIN_PATH, True, None, graph_embeddings)
    elif model == 'BertMF':
        data = CreateDataloader(args, None, None, MAIN_PATH, True, None, None)
    elif model == 'GraphMF':
        graph_embeddings = np.load(main_path + '/glee_embeddings.npy')
        data = CreateDataloader(args, None, None, MAIN_PATH, False, None, graph_embeddings)
    elif model == 'NeuMF':
        data = CreateDataloader(args, None, None, MAIN_PATH, False, None, None)
    return data

def predict(model, user, item, tokenization, graph_embeddings):
    if tokenization is not None and graph_embeddings is not None:
        prediction = model(user, item, tokenization, graph_embeddings)
    elif tokenization is not None:
        prediction = model(user, item, tokenization)
    elif graph_embeddings is not None:
        prediction = model(user, item, graph_embeddings)
    else:
        prediction = model(user, item)
    return prediction


def getModel(model_name, args, num_users, num_items):
    if model_name == 'MultiMF':
        model = MultiMF(args, num_users, num_items, False)
    elif model_name == 'BertMF':
        model = BertMF(args, num_users, num_items)
    elif model_name == 'GraphMF':
        model = GraphMF(args, num_users, num_items)
    elif model_name == 'NeuMF':
        model = NeuMF(args, num_users, num_items)
    return model

DATASETS = {
    'MOOCCubeX': {
        'train_dataset_file': '/train.feather',
        'test_dataset_file': '/test.feather'
    }, 
    'DBLP_v12': {
        'train_dataset_file': '/papers_train.feather',
        'test_dataset_file': '/papers_test.feather'
    }
}

NUM_USER_AND_ITEMS = {
    'MOOCCubeX': (694530, 4701),
    'DBLP_v12': (2794155, 2942027)
}

if __name__ == '__main__':

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
                        default=512,
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
                        default=64,
                        help="predictive factors numbers in the model")
    parser.add_argument("--layers",
                        nargs='+',
                        default=[512, 256, 128, 64, 32, 16, 8],
                        help="MLP layers. Note that the first layer is the concatenation of user \
        and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test",
                        type=int,
                        default=50,
                        help="Number of negative samples for test set")
    parser.add_argument("--token_size",
                        type=int,
                        default=64,
                        help="size of the max token size")
    parser.add_argument("--bert_path",
                        default='./transformers/BERT/model/',
                        help="path to the bert model")
    parser.add_argument("--train_bert",
                        default=False,
                        help="train bert")
    parser.add_argument("--dataset",
                        default='DBLP_v12',
                        help="Dataset")
    parser.add_argument("--model",
                        default='MultiMF',
                        help="Model")

    # set device and parameters
    args = parser.parse_args()
    print(args.dataset, args.model)
    
    DATASET_NAME = args.dataset
    MODEL_NAME = args.model
    TRAIN_DATASET_FILE = DATASETS[DATASET_NAME]['train_dataset_file']
    TEST_DATASET_FILE = DATASETS[DATASET_NAME]['test_dataset_file']
    MAIN_PATH = f'./data/{DATASET_NAME}'
    TRAIN_DATA_PATH = MAIN_PATH + TRAIN_DATASET_FILE
    TEST_DATA_PATH = MAIN_PATH + TEST_DATASET_FILE
    MODEL_PATH = f'./models/{DATASET_NAME}/'
    MODEL = f'{DATASET_NAME}-{MODEL_NAME}'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    writer = SummaryWriter()

    # seed for Reproducibility
    seed_everything(args.seed)

    # load data and construct the train and test datasets
    data = create_dataloader(MODEL_NAME, MAIN_PATH)
    print('Create Train Data Loader')
    train_loader = data.get_train_instance()

    # set model and loss, optimizer
    num_users, num_items = NUM_USER_AND_ITEMS[DATASET_NAME]
    model = getModel(MODEL_NAME, args, num_users, num_items)
    model = model.to(device)
    print(model)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train, evaluation
    best_hr = 0
    for epoch in range(1, args.epochs + 1):
        model.train()  # Enable dropout (if have).
        start_time = time.time()

        for user, item, label, tokenization, graph_embeddings in tqdm(train_loader):
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            if tokenization is not None:
                tokenization = tokenization.to(device)
            if graph_embeddings is not None:
                graph_embeddings = graph_embeddings.to(device)

            optimizer.zero_grad()
            prediction = predict(model, user, item, tokenization, graph_embeddings)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
        print('epoch time: {:.4f}s'.format(time.time() - start_time))

        model.eval()

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model,
               '{}{}.pth'.format(MODEL_PATH, MODEL))

    print('Train done')