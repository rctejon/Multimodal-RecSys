from torch import nn
import torch

class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_users, num_items, factor_num, layers):
        super(MultiLayerPerceptron, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = factor_num
        self.layers = layers

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.fc_layers = nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = nn.ReLU()(vector)
            vector = nn.Dropout(p=0.1)(vector)
        return vector

    def init_weight(self):
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()