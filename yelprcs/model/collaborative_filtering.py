from numpy.lib.arraysetops import isin
from .build import MODEL_REGISTRY
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Linear

class CollaborativeFiltering(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_size = cfg.MODEL.CF.LATENT_SIZE
        self.user_size = cfg.STATISTICS.USER_SIZE
        self.item_size = cfg.STATISTICS.ITEM_SIZE
        self.user_hidden_emb = Parameter(torch.zeros(self.user_size, self.latent_size),
                            requires_grad=True)
        self.item_hidden_emb = Parameter(torch.zeros(self.item_size, self.latent_size),
                            requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.user_hidden_emb)
        torch.nn.init.kaiming_uniform_(self.item_hidden_emb)

        self.user_subnet = None
        self.item_subnet = None
        self.is_training = cfg.IS_TRAIN
        self.loss_func = nn.MSELoss(reduction='mean')

    def process_batch(self, user_batch, item_batch):
        return NotImplementedError

    def extract_batch(self, batched_inputs):
        user_id_list = []
        item_id_list = []
        rating_gt = []
        mean_list = []
        for inputs in batched_inputs:
            user_id_list.append(inputs['user_id'])
            item_id_list.append(inputs['business_id'])
            if 'stars' in inputs:
                rating_gt.append(inputs['stars'])
            mean_list.append(inputs['whole_mean'] + inputs['user_mean'] + inputs['business_mean'])

        return user_id_list, item_id_list, rating_gt, mean_list

    def forward(self, batched_inputs):

        user_id_list, item_id_list, rating_gt, mean_list = self.extract_batch(batched_inputs)

        user_batch = self.user_hidden_emb[user_id_list]
        item_batch = self.item_hidden_emb[item_id_list]

        rating_matrix = self.process_batch(user_batch, item_batch).reshape(-1,1)

        if self.is_training:
            rating_gt = ((torch.tensor(rating_gt) - torch.tensor(mean_list)) / 5).reshape(-1,1)
            loss = self.losses(rating_gt, rating_matrix)
            return loss
        else:
            return rating_matrix

    def losses(self, gt, pred):
        gt = gt.to(device=pred.device)
        loss =  self.loss_func(gt, pred)
        return {'loss': loss}

@MODEL_REGISTRY.register()
class NaiveCF(CollaborativeFiltering):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.user_subnet = lambda x : x
        self.item_subnet = lambda x : x

    def process_batch(self, user_emb_batch, item_emb_batch):
        user_batch = self.user_subnet(user_emb_batch)
        item_batch = self.item_subnet(item_emb_batch)

        rating_matrix = torch.diagonal((user_batch.matmul(item_batch.transpose(1,0))).tanh())

        return rating_matrix

@MODEL_REGISTRY.register()
class SharedCF(NaiveCF):
    def __init__(self, cfg):
        super().__init__(cfg)

        num_fc = cfg.SHARED_CF.NUM_FC

        user_subnet = []
        item_subnet = []
        for _ in range(num_fc):
            user_subnet.append(
                Linear(self.latent_size, self.latent_size)
            )
            user_subnet.append(nn.BatchNorm1d(self.latent_size))
            user_subnet.append(nn.ReLU())

            item_subnet.append(
                Linear(self.latent_size, self.latent_size)
            )
            item_subnet.append(nn.BatchNorm1d(self.latent_size))
            item_subnet.append(nn.ReLU())

        self.user_subnet = nn.Sequential(*user_subnet)
        self.item_subnet = nn.Sequential(*item_subnet)

        # Initialization
        for modules in [self.user_subnet, self.item_subnet]:
            for layer in modules.modules():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight)

@MODEL_REGISTRY.register()
class AttentiveCF(SharedCF):
    def __init__(self, cfg):
        super().__init__(cfg)

        user_attention = {}
        item_attention = {}

        attention_attrib = ['key', 'value', 'query']

        for attrib in attention_attrib:
            for attention_subnet in [user_attention, item_attention]:
                attention_subnet[attrib] = []
                attention_subnet[attrib].append(
                    Linear(self.latent_size, self.latent_size)
                )
                attention_subnet[attrib].append(nn.BatchNorm1d(self.latent_size))
                attention_subnet[attrib].append(nn.ReLU())

                for layer in attention_subnet[attrib]:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.kaiming_uniform_(layer.weight)

        self.user_attention = user_attention
        self.item_attention = item_attention

    def process_batch(self, user_emb_batch, item_emb_batch):
        user_batch = self.user_subnet(user_emb_batch)
        item_batch = self.item_subnet(item_emb_batch)

        attention_batch = []
        for attention_subnet, batch in zip([self.user_attention, self.item_attention], [user_batch, item_batch]):
            key_batch = attention_subnet['key'](batch)
            value_batch = attention_subnet['value'](batch)
            query_batch = attention_subnet['query'](batch)

            attention = key_batch * query_batch.transpose(1,0)
            attention_batch.append(attention * value_batch)

        rating_matrix = (attention_batch[0] * attention_batch[1]).sigmoid()

        return rating_matrix
