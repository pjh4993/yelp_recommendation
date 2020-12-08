from numpy.lib.arraysetops import isin
from .build import MODEL_REGISTRY
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Linear
import numpy as np
import itertools

class CollaborativeFiltering(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_size = cfg.MODEL.CF.LATENT_SIZE
        self.user_size = cfg.STATISTICS.USER_SIZE
        self.item_size = cfg.STATISTICS.ITEM_SIZE
        self.user_hidden_emb = nn.Embedding(self.user_size, self.latent_size)
        self.item_hidden_emb = nn.Embedding(self.item_size, self.latent_size)

        """
        self.user_hidden_emb = Parameter(torch.zeros(self.user_size, self.latent_size),
                            requires_grad=False)
        self.item_hidden_emb = Parameter(torch.zeros(self.item_size, self.latent_size),
                            requires_grad=False)

        mean = torch.tensor(2.0 / self.latent_size).sqrt()
        std = mean  / 10
        torch.nn.init.normal_(self.user_hidden_emb, mean, std)
        torch.nn.init.normal_(self.item_hidden_emb, mean, std)
        torch.nn.init.kaiming_normal_(self.user_hidden_emb)
        torch.nn.init.kaiming_normal_(self.item_hidden_emb)

        """
        self.user_subnet = None
        self.item_subnet = None
        self.loss_func = getattr(nn, cfg.MODEL.CF.LOSS_FUNC)(reduction="none")
        #self.loss_func = nn.MSELoss(reduction='mean')
        #self.loss_func = nn.SmoothL1Loss(reduction='mean')
        #self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    @property
    def device(self):
        return self.user_hidden_emb.weight.device

    def extract_batch(self, batched_inputs):
        idx_tensor = torch.LongTensor(batched_inputs)[:,[0,2]].to(self.device)
        rating_gt = torch.LongTensor(batched_inputs)[:,1].to(self.device) - 1
        mean_tensor = torch.clip(torch.FloatTensor(batched_inputs)[:,3].to(self.device) ,0 , 5)

        return idx_tensor, mean_tensor, rating_gt

    def forward(self, batched_inputs):
        batched_inputs = list(itertools.chain(*batched_inputs))
        idx_tensor, mean_tensor, rating_gt = self.extract_batch(batched_inputs)

        user_batch = self.user_hidden_emb(idx_tensor[:,0])
        item_batch = self.item_hidden_emb(idx_tensor[:,1])

        rating_matrix = self.process_batch(user_batch, item_batch, mean_tensor)
        #print(rating_matrix[:10].flatten())

        if self.training:
            loss = self.losses(rating_gt, rating_matrix)
            with torch.no_grad():
                eval = self.mini_eval(rating_gt, rating_matrix)
            return loss , eval
        else:
            output = self.inference(rating_matrix)
            return output.cpu().numpy()


    def process_batch(self, user_batch, item_batch):
        return NotImplementedError

    def mini_eval(self, gt, pred):
        return NotImplementedError

    def losses(self, gt, pred):
        return NotImplementedError
    
    def inference(self, pred):
        return NotImplementedError

@MODEL_REGISTRY.register()
class NaiveCF(CollaborativeFiltering):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.user_subnet = lambda x : x
        self.item_subnet = lambda x : x
        self.relation_net = nn.Linear(cfg.MODEL.CF.LATENT_SIZE, 1)
        torch.nn.init.kaiming_normal_(self.relation_net.weight)
        torch.nn.init.constant_(self.relation_net.bias, 0.0)

    def process_batch(self, user_emb_batch, item_emb_batch, mean_tensor):
        user_batch = self.user_subnet(user_emb_batch)
        item_batch = self.item_subnet(item_emb_batch)
        #rating_matrix = torch.diagonal(user_batch.matmul(item_batch.t()))
        rating_matrix = self.relation_net(user_batch * item_batch).sigmoid() * 4
        return rating_matrix
    
    def mini_eval(self, gt, pred):
        rmse = ((gt - self.inference(pred)) ** 2).mean().sqrt()
        return rmse.cpu().item()

    def losses(self, gt, pred):
        gt = gt.float().to(device=pred.device)
        gt = gt.reshape(-1, 1)
        pred = pred.reshape(-1, 1)
        #print(gt.max(), gt.min(), pred.max(), pred.min())
        loss = (self.loss_func(pred, gt)).mean()
        #loss = ((gt - self.inference(pred)) ** 2).mean()
        return {'loss': loss}
    
    def inference(self, pred):
        return torch.clip(pred, 0 , 4)


@MODEL_REGISTRY.register()
class RelationCF(CollaborativeFiltering):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.user_subnet = lambda x : x
        self.item_subnet = lambda x : x
        self.relation_net = nn.Linear(
            cfg.MODEL.CF.LATENT_SIZE * 2, 5
        )
        torch.nn.init.kaiming_uniform_(self.relation_net.weight)
        torch.nn.init.constant_(self.relation_net.bias, 0)

    def process_batch(self, user_emb_batch, item_emb_batch, mean_tensor):
        user_batch = self.user_subnet(user_emb_batch)
        item_batch = self.item_subnet(item_emb_batch)
        rating_matrix = torch.cat([user_batch, item_batch], dim=1)
        rating_matrix = self.relation_net(rating_matrix)

        return rating_matrix
    
    def mini_eval(self, gt, pred):
        acc = (gt == self.inference(pred)).sum() / len(gt)
        return acc.cpu().item()

    def losses(self, gt, pred):
        gt = gt.to(device=pred.device)
        #weight = (1 - nn.Softmax(dim=1)(pred)[torch.arange(len(gt)), gt]).detach()
        loss =  (self.loss_func(pred, gt)).mean()
        return {'loss': loss}

    def inference(self, pred):
        return torch.argmax(nn.Softmax(dim=1)(pred), dim=1)


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
            user_subnet.append(nn.BatchNorm1d(self.latent_size))
            user_subnet.append(nn.ReLU())

            item_subnet.append(
                Linear(self.latent_size, self.latent_size)
            )
            item_subnet.append(nn.BatchNorm1d(self.latent_size))
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
