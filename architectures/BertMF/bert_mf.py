from torch import nn
import torch
from architectures.mlp import MultiLayerPerceptron
from architectures.generalized_matrix_factorization import GeneralizedMatrixFactorization


class BertMF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(BertMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp =  int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout
        self.token_size = args.token_size
        self.train_bert = args.train_bert

        self.mlp = MultiLayerPerceptron(num_users, num_items, self.factor_num_mlp, self.layers)
        self.gmf = GeneralizedMatrixFactorization(num_users, num_items, self.factor_num_mf)
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', args.bert_path)

        if not self.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf + self.bert.config.hidden_size, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        self.mlp.init_weight()
        self.gmf.init_weight()

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices, tokenizations):
        mlp_vector = self.mlp(user_indices, item_indices)
        mf_vector = self.gmf(user_indices, item_indices)
        if self.train_bert:
            bert_vector = self.bert(**tokenizations).pooler_output
        else:
            bert_vector = tokenizations

        vector = torch.cat([mlp_vector, mf_vector, bert_vector], dim=1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()