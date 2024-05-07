from torch import nn
import torch

class GeneralizedMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, factor_num):
        super(GeneralizedMatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = factor_num

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)

        return element_product

    def init_weight(self):
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()