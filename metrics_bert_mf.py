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
        default=32,
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

    text_column = 'embedding' if not args.train_bert else 'tokenization'

    do_precalc = not os.path.exists(f'{MAIN_PATH}/test_tokenizations_{args.num_ng_test}_{args.token_size}.pkl') or not os.path.exists(f'{MAIN_PATH}/train_tokenizations_{args.num_ng}_{args.token_size}.pkl')

    if not args.train_bert:
        do_precalc = not os.path.exists(f'{MAIN_PATH}/test_embeddings_{args.num_ng}_{args.token_size}.pkl') or not os.path.exists(f'{MAIN_PATH}/train_embeddings_{args.num_ng}_{args.token_size}.pkl')

    if do_precalc:

        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './transformers/BERT/tokenizer/')
        bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', args.bert_path)

        # seed for Reproducibility
        seed_everything(args.seed)

        # load data

        text_df = create_text_df(tokenizer=tokenizer, max_length=args.token_size, train_bert=args.train_bert, model=bert_model, device=device)
        default_row = tokenize('Description: ', tokenizer, args.token_size)

        if not args.train_bert:
            default_row = get_bert_embedding('Description: ', tokenizer, bert_model, args.token_size, device)
        train_rating_data = pd.read_feather(TRAIN_DATA_PATH)
        train_rating_data = train_rating_data.merge(text_df, how='left', on='course_id')
        train_rating_data[text_column] = train_rating_data[text_column].apply(lambda x: default_row if type(x) == float else x)
        train_rating_data = train_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})
        


        test_rating_data = pd.read_feather(TEST_DATA_PATH)
        test_rating_data = test_rating_data.merge(text_df, how='left', on='course_id')
        test_rating_data[text_column] = test_rating_data[text_column].apply(lambda x: default_row if type(x) == float else x)
        test_rating_data = test_rating_data.rename(columns={'id': 'user_id', 'course_id': 'item_id'})
        

        ratings = pd.concat([train_rating_data, test_rating_data], ignore_index=True)

        train_rating_data = _reindex(train_rating_data)
        test_rating_data = _reindex(test_rating_data)

        
        tokenizations = None
        tokenizations = ratings[['item_id', text_column]].drop_duplicates(subset=['item_id'])
        tokenizations.set_index('item_id', inplace=True)
    else:
        train_rating_data = None
        test_rating_data = None
        tokenizations = None

    # construct the train and test datasets

    data = CreateDataloader(args, train_rating_data, test_rating_data, MAIN_PATH, True, tokenizations)
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
