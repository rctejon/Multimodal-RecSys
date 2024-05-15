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
from architectures.BertMF.bert_mf import BertMF
from loaders.create_dataloader import CreateDataloader
from tqdm import tqdm
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def tokenize(text, tokenizer, max_length):
    """
    Tokenize the text
    """
    tokenization = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

    tokenization = {
        'input_ids': tokenization['input_ids'],
        'attention_mask': tokenization['attention_mask'],
        'token_type_ids': tokenization['token_type_ids']
    }
    return tokenization

def get_bert_embedding(text, tokenizer, model, max_length=32, device='cuda:0'):
    """
    Get the bert embeddings
    """
    model.to(device)
    tokenization = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        embedding = model(**tokenization).pooler_output.to('cpu')
    return embedding


def create_text_df(tokenizer, max_length, train_bert, model = None, device='cuda:0'):
    """
    Create a dataframe from the text dictionary
    """
    course_texts = pickle.load(open('./pickles/course_texts.pkl', 'rb'))
    text_df = pd.DataFrame.from_dict(course_texts, orient='index')
    text_df.reset_index(inplace=True)
    text_df.columns = ['course_id', 'text']
    text_df['course_id'] = text_df['course_id'].apply(lambda x: x.split('_')[-1])
    text_df['course_id'] = text_df['course_id'].astype(int)
    text_df['text'] = text_df['text'].astype(str)
    if not train_bert:
        text_df['embedding'] = text_df['text'].apply(lambda x: get_bert_embedding(x, tokenizer, model, max_length, device))
    else:
        text_df['tokenization'] = text_df['text'].apply(lambda x: tokenize(x, tokenizer, max_length))
    text_df.drop(columns=['text'], inplace=True)
    return text_df

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
    MODEL_NAME = 'BertMF'
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
        default=2,
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
        help="train bert")
    
    # set device and parameters
    args = parser.parse_args()
    print(args.epochs, args.lr, args.dropout, args.batch_size, args.factor_num, args.layers, args.num_ng, args.num_ng_test, args.top_k, args.train_bert)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    writer = SummaryWriter()

    text_column = 'embedding' if not args.train_bert else 'tokenization'

    # seed for Reproducibility
    seed_everything(args.seed)

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './transformers/BERT/tokenizer/')
    bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', args.bert_path)

    # load data
    print(TRAIN_DATA_PATH)
    train_rating_data = pd.read_feather(TRAIN_DATA_PATH)
    test_rating_data = pd.read_feather(TEST_DATA_PATH)
    print(train_rating_data.head())
    print(train_rating_data.dtypes)
    print('Begin Tokenization')
    text_df = create_text_df(tokenizer=tokenizer, max_length=args.token_size, train_bert=args.train_bert, model=bert_model, device=device)
    print(text_df.head())
    print(text_df.dtypes)

    train_rating_data = train_rating_data.merge(text_df, how='left', on='course_id')
    test_rating_data = test_rating_data.merge(text_df, how='left', on='course_id')

    default_row = tokenize('Description: ', tokenizer, args.token_size)

    if not args.train_bert:
        default_row = get_bert_embedding('Description: ', tokenizer, bert_model, args.token_size, device)

    print(default_row)

    train_rating_data[text_column] = train_rating_data[text_column].apply(lambda x: default_row if type(x) == float else x)
    test_rating_data[text_column] = test_rating_data[text_column].apply(lambda x: default_row if type(x) == float else x)

    print(train_rating_data.head())

    train_rating_data = train_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})
    test_rating_data = test_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})

    ratings = pd.concat([train_rating_data, test_rating_data], ignore_index=True)

    tokenizations = None

    if not args.train_bert:
        if not os.path.exists(f'{MAIN_PATH}/test_embeddings_{args.num_ng_test}_{args.token_size}.pkl') or not os.path.exists(f'{MAIN_PATH}/train_embeddings_{args.num_ng}_{args.token_size}.pkl'):
            tokenizations = ratings[['item_id', 'embedding']].drop_duplicates(subset=['item_id'])
            tokenizations.set_index('item_id', inplace=True)

    else:
        if not os.path.exists(f'{MAIN_PATH}/test_tokenizations_{args.num_ng_test}_{args.token_size}.pkl') or not os.path.exists(f'{MAIN_PATH}/train_tokenizations_{args.num_ng}_{args.token_size}.pkl'):
            tokenizations = ratings[['item_id', 'tokenization']].drop_duplicates(subset=['item_id'])
            tokenizations.set_index('item_id', inplace=True)

    # set the num_users, items
    num_users = ratings['user_id'].nunique()+1
    num_items = ratings['item_id'].nunique()+1

    print(num_users, num_items)

    train_rating_data = _reindex(train_rating_data)
    test_rating_data = _reindex(test_rating_data)


    # construct the train and test datasets

    data = CreateDataloader(args, train_rating_data, test_rating_data, MAIN_PATH, True, tokenizations)
    print('Create Train Data Loader')
    train_loader = data.get_train_instance()

    # set model and loss, optimizer
    model = BertMF(args, num_users, num_items)
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

        for user, item, label, tokenization in tqdm(train_loader):
            # print(user.size(), item.size(), label.size())
            # print(user, item, label)
            
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            tokenization = tokenization.to(device)

            optimizer.zero_grad()
            # print('Zero Grad')
            prediction = model(user, item, tokenization)
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