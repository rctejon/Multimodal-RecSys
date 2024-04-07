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
    train = ratings.loc[ratings['rank_latest'] > 1]
    assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
    return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

if __name__ == '__main__':
    DATASET_NAME = 'ml-1m'
    MODEL_NAME = 'NeuMF'
    DATASET_FILE = 'ratings.dat'
    MAIN_PATH = f'./data/{DATASET_NAME}/'
    DATA_PATH = MAIN_PATH + DATASET_FILE
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
    
    # set device and parameters
    args = parser.parse_args()
    print(args.epochs, args.lr, args.dropout, args.batch_size, args.factor_num, args.layers, args.num_ng, args.num_ng_test, args.top_k, args.out)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    writer = SummaryWriter()

    # seed for Reproducibility
    seed_everything(args.seed)

    # load data
    ml_1m = pd.read_csv(
        DATA_PATH,
        sep="::",
        names = ['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python')

    # set the num_users, items
    num_users = ml_1m['user_id'].nunique()+1
    num_items = ml_1m['item_id'].nunique()+1

    ml_1m = _reindex(ml_1m)
    train_ml_1m, test_ml_1m = _leave_one_out(ml_1m)

    # construct the train and test datasets
    data = CreateDataloader(args, train_ml_1m, test_ml_1m)
    train_loader = data.get_train_instance()
    test_loader = data.get_test_instance()

    # set model and loss, optimizer
    model = NeuMF(args, num_users, num_items)
    model = model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # train, evaluation
    best_hr = 0
    for epoch in range(1, args.epochs+1):
        model.train() # Enable dropout (if have).
        start_time = time.time()

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/Train_loss', loss.item(), epoch)

        model.eval()
        HR, NDCG = metrics(model, test_loader, args.top_k, device)
        writer.add_scalar('Perfomance/HR@10', HR, epoch)
        writer.add_scalar('Perfomance/NDCG@10', NDCG, epoch)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(MODEL_PATH):
                    os.mkdir(MODEL_PATH)
                torch.save(model,
                    '{}{}.pth'.format(MODEL_PATH, MODEL))

    writer.close()

    print("Best epoch {:03d}: HR@10 = {:.3f}, NDCG@10 = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
