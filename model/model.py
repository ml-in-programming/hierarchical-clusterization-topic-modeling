import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class SkipGramModel(BaseModel):
    """ word2vec model """

    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed_centers = nn.Embedding(vocab_size, embed_size)
        self.embed_contexts = nn.Embedding(vocab_size, embed_size)

    def forward(self, centers, pos_contexts, neg_contexts):
        batch_size = centers.size()[0]

        centers = self.embed_centers(centers)
        pos_contexts = self.embed_contexts(pos_contexts)
        neg_contexts = self.embed_contexts(neg_contexts)

        pos_mm = torch.bmm(centers.unsqueeze(1), pos_contexts.unsqueeze(2)).squeeze()
        pos_loss = F.logsigmoid(pos_mm).sum()

        neg_mm = torch.bmm(neg_contexts, centers.unsqueeze(2)).squeeze().sum(dim=1)
        neg_loss = F.logsigmoid(-neg_mm).sum()

        return -(pos_loss + neg_loss) / batch_size