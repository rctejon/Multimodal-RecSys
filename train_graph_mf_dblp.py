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
from architectures.GraphMF.graph_mf import GraphMF
from loaders.create_dataloader import CreateDataloader
from tqdm import tqdm
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
    DATASET_NAME = 'DBLP_v12'
    MODEL_NAME = 'GraphMF'
    TRAIN_DATASET_FILE = 'papers_train.feather'
    TEST_DATASET_FILE = 'papers_test.feather'
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
        default=64,
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
    
    # set device and parameters
    args = parser.parse_args()
    print(args.epochs, args.lr, args.dropout, args.batch_size, args.factor_num, args.layers, args.num_ng, args.num_ng_test, args.top_k, args.train_bert)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    writer = SummaryWriter()

    # text_column = 'embedding' if not args.train_bert else 'tokenization'

    # # seed for Reproducibility
    # seed_everything(args.seed)

    # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './transformers/BERT/tokenizer/')
    # bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', args.bert_path)

    # # load data
    # print(TRAIN_DATA_PATH)
    # train_rating_data = pd.read_feather(TRAIN_DATA_PATH)
    # test_rating_data = pd.read_feather(TEST_DATA_PATH)
    # train_rating_data = train_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})
    # test_rating_data = test_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})

    # ratings = pd.concat([train_rating_data, test_rating_data], ignore_index=True)
    # # set the num_users, items
    # num_users = ratings['user_id'].nunique()+1
    # num_items = ratings['item_id'].nunique()+1

    # print(num_users, num_items)

    train_rating_data = None
    test_rating_data = None
    tokenizations = None

    graph_embeddings = np.load(MAIN_PATH + 'glee_embeddings.npy')
    print(graph_embeddings.shape)


    # construct the train and test datasets

    data = CreateDataloader(args, train_rating_data, test_rating_data, MAIN_PATH, False, tokenizations, graph_embeddings, False)
    print('Create Train Data Loader')
    train_loader = data.get_train_instance()

    # set model and loss, optimizer
    model = GraphMF(args, 2794155, 2942027, False)
    # model = torch.load('{}{}.pth'.format(MODEL_PATH, MODEL))
    model = model.to(device)
    print(model)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # train, evaluation
    best_hr = 0
    for epoch in range(1, args.epochs+1):
        model.train() # Enable dropout (if have).
        start_time = time.time()

        for user, item, label, _, graph_embeddings in tqdm(train_loader):
            # print(user.size(), item.size(), label.size())
            # print(user, item, label)
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            graph_embeddings = graph_embeddings.to(device)


            optimizer.zero_grad()
            # print('Zero Grad')
            prediction = model(user, item, graph_embeddings)
            # print('Prediction')
            loss = loss_function(prediction, label)
            # print('Loss')
            loss.backward()
            # print('Backward')
            optimizer.step()
            # print('Step')

        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
        print('epoch time: {:.4f}s'.format(time.time()-start_time))
        # if loss.item() < 0.001:
        #     break

        model.eval()

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model,
        '{}{}.pth'.format(MODEL_PATH, MODEL))

    print('Train done')