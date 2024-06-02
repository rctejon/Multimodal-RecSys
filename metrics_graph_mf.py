import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tensorboardX import SummaryWriter
from metrics.metrics import metrics
from architectures.NeuMF.neu_mf import NeuMF
from loaders.create_dataloader import CreateDataloader
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    MODEL_NAME = 'GraphMF'
    TRAIN_DATASET_FILE = 'train.feather'
    TEST_DATASET_FILE = 'test.feather'
    MAIN_PATH = f'./data/{DATASET_NAME}/'
    TRAIN_DATA_PATH = MAIN_PATH + TRAIN_DATASET_FILE
    TEST_DATA_PATH = MAIN_PATH + TEST_DATASET_FILE
    MODEL = f'{DATASET_NAME}-{MODEL_NAME}'
    # MODEL = f'1_epoch'
    MODEL_PATH = f'./models/{DATASET_NAME}/{MODEL}.pth'

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
        default=[128,64,32,16,8],
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
        help="train the bert model")
    
    # set device and parameters
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    writer = SummaryWriter()
    
    train_rating_data = None
    test_rating_data = None
    tokenizations = None

    graph_embeddings = np.load(MAIN_PATH + 'glee_embeddings.npy')
    print(graph_embeddings.shape)

    # construct the train and test datasets

    data = CreateDataloader(args, train_rating_data, test_rating_data, MAIN_PATH, False, tokenizations, graph_embeddings)
    print('Create Test Data Loader')
    test_loader = data.get_test_instance()

    start_time = time.time()

    # set model and loss, optimizer
    model = torch.load(MODEL_PATH)
    model = model.to(device)

    top_ks = [1, 3, 5, 10]

    print('Calculate Metrics')
    HR, NDCG, MRR, RECALL, PRECISION = metrics(model, test_loader, top_ks, device, args.num_ng_test)

    print(f"MRR: {MRR}")

    for top_k in top_ks:
        print(f"HR@{top_k}: {HR[top_k]}\tNDGC@{top_k}: {NDCG[top_k]}\tRECALL@{top_k}: {RECALL[top_k]}\tPRECISION@{top_k}: {PRECISION[top_k]}")
    writer.close()
