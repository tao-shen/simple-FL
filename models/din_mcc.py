from .model_base import *
from .model_fn import *


class DIN_mcc(nn.Module, Model_fn):
    def __init__(self, args):
        super(DIN_mcc, self).__init__()

        self.args = args
        self.features = args.features
        self._estimator_type = 'classifier'
        self.num_inputs = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        self.cat_embeddings = nn.ModuleDict()
        self.seq_embeddings = nn.ModuleDict()
        cat_size = 0

        for embed_key in args.embedding.keys():
            self.embeddings[embed_key] = nn.Embedding(
                args.embedding[embed_key]['num'], args.embedding[embed_key]['size'])
            for feats_key, feats_value in args.use_feats.items():
                if embed_key in feats_key:
                    if feats_value == 'cat_feats':
                        self.cat_embeddings[feats_key] = self.embeddings[embed_key]
                    if feats_value == 'seq_feats':
                        self.seq_embeddings[feats_key] = self.embeddings[embed_key]
                    cat_size += args.embedding[embed_key]['size']
        args.item_embed_size = sum(
            [v['size'] for k, v in args.embedding.items() if 'item' in k])

        for key in self.features['num_feats']:
            self.num_inputs[key] = nn.Identity()
            cat_size += 1

        self.pooling = Pooling('attention', dim=1, args=args)
        self.mlp = MLP(cat_size, self.args)

        Model_fn.__init__(self, args)

    def forward(self, inputs):

        embedded = {}
        for key, module in self.num_inputs.items():
            out = module(inputs[key]).unsqueeze(-1)
            embedded[key] = out
        can_embedded, exp_embedded, ipv_embedded = [], [], []
        for key, module in self.cat_embeddings.items():
            out = module(inputs[key])
            if 'cand_item' in key:
                can_embedded.append(out)
            else:
                embedded[key] = out
        embedded['cand_item'] = torch.cat(can_embedded, dim=1)
        for key, module in self.seq_embeddings.items():
            seq_out = module(inputs[key])
            if 'exp_item' in key:
                exp_embedded.append(seq_out)
            elif 'ipv_item' in key:
                ipv_embedded.append(seq_out)

        exp_seq = torch.cat(exp_embedded, dim=-1)
        exp_out = self.pooling(exp_seq, embedded['cand_item'])
        embedded['exp_item'] = exp_out

        ipv_seq = torch.cat(ipv_embedded, dim=-1)
        ipv_out = self.pooling(ipv_seq, embedded['cand_item'])
        embedded['ipv_item'] = ipv_out

        emb_cat = torch.cat(list(embedded.values()), dim=1)
        score_logits = -torch.log(1/inputs['score'].unsqueeze(-1)-1)
        output = torch.sigmoid(self.mlp(emb_cat)+score_logits)
        return output
