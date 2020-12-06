from numpy.lib.arraysetops import isin
from .build import MODEL_REGISTRY
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Linear
import numpy as np

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

        torch.nn.init.kaiming_normal_(self.user_hidden_emb)
        torch.nn.init.kaiming_normal_(self.item_hidden_emb)

        self.user_subnet = None
        self.item_subnet = None
        self.loss_func = nn.MSELoss(reduction='mean')
        #self.loss_func = nn.SmoothL1Loss(reduction='mean')

    @property
    def device(self):
        return self.user_hidden_emb.device
    def process_batch(self, user_batch, item_batch):
        return NotImplementedError

    def extract_batch(self, batched_inputs):
        idx_tensor = torch.LongTensor(batched_inputs)[:,[0,2]].to(self.device)
        rating_gt = torch.FloatTensor(batched_inputs)[:,1].to(self.device)
        mean_tensor = torch.clip(torch.FloatTensor(batched_inputs)[:,3].to(self.device) ,0 , 5)

        return idx_tensor, mean_tensor, rating_gt

    def forward(self, batched_inputs):

        idx_tensor, mean_tensor, rating_gt = self.extract_batch(batched_inputs)

        user_batch = self.user_hidden_emb[idx_tensor[:,0]]
        item_batch = self.item_hidden_emb[idx_tensor[:,1]]

        rating_matrix = self.process_batch(user_batch, item_batch)

        if self.training:
            loss = self.losses(rating_gt.reshape(-1, 1), rating_matrix.reshape(-1,1))
            return loss
        else:
            output = torch.clip(rating_matrix, 0, 5)
            return output.cpu().numpy()


    def losses(self, gt, pred):
        gt = gt.to(device=pred.device)
        print(pred.max(), gt.max(), pred.min(), gt.min())
        loss =  self.loss_func(pred, gt)
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

        rating_matrix = torch.sum(item_batch * user_batch, dim=1)

        return rating_matrix

@MODEL_REGISTRY.register()
class SharedCF(NaiveCF):
    def __init__(self, cfg):
        super().__init__(cfg)

        num_fc = cfg.MODEL.SHARED_CF.NUM_FC

        user_subnet = []
        item_subnet = []
        for _ in range(num_fc):
            user_subnet.append(
                Linear(self.latent_size, self.latent_size)
            )
            #user_subnet.append(nn.BatchNorm1d(self.latent_size))
            user_subnet.append(nn.ReLU())

            item_subnet.append(
                Linear(self.latent_size, self.latent_size)
            )
            #item_subnet.append(nn.BatchNorm1d(self.latent_size))
            item_subnet.append(nn.ReLU())

        user_subnet.append(
            Linear(self.latent_size, self.latent_size)
        )
        item_subnet.append(
            Linear(self.latent_size, self.latent_size)
        )

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
                #attention_subnet[attrib].append(nn.BatchNorm1d(self.latent_size))
                attention_subnet[attrib].append(nn.ReLU())

                attention_subnet[attrib].append(
                    Linear(self.latent_size, self.latent_size)
                )

                for layer in attention_subnet[attrib]:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.kaiming_uniform_(layer.weight)
                
                attention_subnet[attrib] = nn.Sequential(*attention_subnet[attrib])

        self.user_attention = nn.ModuleDict(user_attention)
        self.item_attention = nn.ModuleDict(item_attention)

    def process_batch(self, user_emb_batch, item_emb_batch):
        user_batch = self.user_subnet(user_emb_batch)
        item_batch = self.item_subnet(item_emb_batch)

        attention_batch = []
        for attention_subnet, batch in zip([self.user_attention, self.item_attention], [user_batch, item_batch]):
            key_batch = attention_subnet['key'](batch)
            value_batch = attention_subnet['value'](batch)
            query_batch = attention_subnet['query'](batch)

            attention = key_batch.matmul(query_batch.transpose(1,0))
            attention_batch.append(attention.matmul(value_batch))

        rating_matrix = torch.sum(attention_batch[0] * attention_batch[1], dim=1).tanh()

        return rating_matrix
