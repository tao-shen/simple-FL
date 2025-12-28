from .model_base import *
from .model_fn import *


class DNN(nn.Module, Model_fn):
    def __init__(self, args):
        super(DNN, self).__init__()

        self.args = args
        self.features = args.features
        self._estimator_type = 'classifier'
        self.num_inputs = nn.ModuleDict()
        self.cat_embeddings = nn.ModuleDict()
        self.seq_embeddings = nn.ModuleDict()
        cat_size = 0

        self.item_embeddings = nn.Embedding(
            args.embedding['item']['num'], args.embedding['item']['size'])
        args.item_embed_size = args.embedding['item']['size']

        for key in self.features['num_feats']:
            self.num_inputs[key] = nn.Identity()
            cat_size += 1
        for key in self.features['cat_feats']:
            if 'item' in key:
                self.cat_embeddings[key] = self.item_embeddings
                cat_size += args.embedding['item']['size']
            else:
                self.cat_embeddings[key] = nn.Embedding(
                    args.embedding[key]['num'], args.embedding[key]['size'])
                cat_size += args.embedding[key]['size']
        for key in self.features['seq_feats']:
            if 'item' in key:
                self.seq_embeddings[key] = self.item_embeddings
                cat_size += args.embedding['item']['size']
            else:
                self.cat_embeddings[key] = nn.Embedding(
                    args.embedding[key]['num'], args.embedding[key]['size'])
                cat_size += args.embedding[key]['size']

        self.pooling = Pooling('mean', dim=1, args=args)
        self.mlp = MLP(cat_size, self.args)
        self.final = nn.Linear(2, 1)

        Model_fn.__init__(self, args)

    def forward(self, inputs):

        embedded = {}
        for key, module in self.num_inputs.items():
            out = module(inputs[key]).unsqueeze(-1)
            embedded[key] = out
        for key, module in self.cat_embeddings.items():
            out = module(inputs[key])
            embedded[key] = out
        for key, module in self.seq_embeddings.items():
            seq_out = module(inputs[key])
            out = self.pooling(seq_out, embedded['cand_item_id'])
            embedded[key] = out

        emb_cat = torch.cat(list(embedded.values()), dim=1)
        output = torch.sigmoid(self.mlp(emb_cat))

        return output
