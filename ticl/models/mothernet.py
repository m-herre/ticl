import torch, wandb
import torch.nn as nn
from torch.nn import TransformerEncoder

from ticl.models.encoders import OneHotAndLinear
from ticl.models.decoders import MLPModelDecoder, GradTreeDecoder
from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.models.encoders import Linear

from ticl.utils import SeqBN, get_init_method


class ModelPredictor(nn.Module):

    def tree_forward(self, x_test, feature_idx, thresholds, L):
        """
        x_test:      (n_test, batch, n_features)
        feature_idx: (batch, n_nodes)          # argmax over I_logits (per node)
        thresholds:  (batch, n_nodes)          # gathered per chosen feature
        L:           (batch, n_leaves, n_out)  # leaf logits

        returns: (n_test, batch, n_out)
        """
        n_test, batch, _ = x_test.shape
        device = x_test.device
        depth = self.tree_depth

        # Start at root (node 0) in global array indexing
        node_idx = torch.zeros((n_test, batch), dtype=torch.long, device=device)

        for _ in range(depth):
            # Pairwise batch indexing to select (feature, threshold) for each (sample, dataset)
            batch_ids = (
                torch.arange(batch, device=device).unsqueeze(0).expand(n_test, -1)
            )
            f = feature_idx[batch_ids, node_idx]  # (n_test, batch)
            t = thresholds[batch_ids, node_idx]  # (n_test, batch)

            # Feature values for each sample/dataset at this node
            x_f = torch.gather(x_test, 2, f.unsqueeze(-1)).squeeze(
                -1
            )  # (n_test, batch)

            # Hard decision: 0 = left, 1 = right
            decision = (x_f > t).long()

            # Move to child in global array indexing: left=2*i+1, right=2*i+2
            node_idx = 2 * node_idx + 1 + decision

        # Convert global node index to leaf index in [0 .. 2^d - 1]
        leaf_base = (1 << depth) - 1
        leaf_idx = node_idx - leaf_base  # (n_test, batch)

        # Gather leaf logits
        batch_ids = torch.arange(batch, device=device).unsqueeze(0).expand(n_test, -1)
        leaf_logits = L[batch_ids, leaf_idx]  # (n_test, batch, n_out)

        return leaf_logits

    def forward(self, src, single_eval_pos=None):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src
        _, x, y = src

        # Encode training part
        x_enc = self.encoder(x)
        if self.y_encoder is None:
            enc_train = x_enc[:single_eval_pos]
        else:
            y_enc = self.y_encoder(
                y.unsqueeze(-1) if len(y.shape) < len(x.shape) else y
            )
            enc_train = x_enc[:single_eval_pos] + y_enc[:single_eval_pos]
        if self.decoder_type in ["special_token", "special_token_simple"]:
            enc_train = torch.cat(
                [self.token_embedding.repeat(1, enc_train.shape[1], 1), enc_train], 0
            )
        elif self.decoder_type == "class_tokens":
            if not isinstance(self.y_encoder, OneHotAndLinear):
                raise ValueError(
                    "class_tokens decoder type is only supported with OneHotAndLinear y_encoder"
                )
            repeated_class_tokens = self.y_encoder.weight.T.unsqueeze(1).repeat(
                1, enc_train.shape[1], 1
            )
            enc_train = torch.cat([repeated_class_tokens, enc_train], 0)

        # Run transformer
        output = self.inner_forward(enc_train)

        if self.child_model == "mlp":
            (b1, w1), *layers = self.decoder(output, y[:single_eval_pos])

            x_test_nona = torch.nan_to_num(x[single_eval_pos:], nan=0)

            h = (x_test_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2)

            if self.decoder.weight_embedding_rank is not None and len(layers):
                h = torch.matmul(h, self.decoder.shared_weights[0])
            h = h + b1

            for i, (b, w) in enumerate(layers):
                if self.predicted_activation == "relu":
                    h = torch.relu(h)
                elif self.predicted_activation == "gelu":
                    h = torch.nn.functional.gelu(h)
                else:
                    raise ValueError(
                        f"Unsupported predicted activation: {self.predicted_activation}"
                    )
                h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                if (
                    self.decoder.weight_embedding_rank is not None
                    and i != len(layers) - 1
                ):
                    # last layer has no shared weights
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
                h = h + b

        elif self.child_model == "gradtree":
            I_logits, T, L = self.decoder(output, y[:single_eval_pos])
            x_test = torch.nan_to_num(x[single_eval_pos:], nan=0)

            # 1. Select features per node
            feature_idx = I_logits.argmax(-1)  # (batch, n_nodes)
            # 2. Gather thresholds of selected features
            thresholds = T.gather(2, feature_idx.unsqueeze(-1)).squeeze(
                -1
            )  # (batch, n_nodes)

            # 3. Forward pass through emitted tree
            h = self.tree_forward(x_test, feature_idx, thresholds, L)

        else:
            raise ValueError(f"Unknown child_model type: {self.child_model}")

        if h.isnan().all():
            print("NAN")
            raise ValueError("NAN")
        return h


class MotherNet(ModelPredictor):
    def __init__(
        self,
        *,
        n_out,
        emsize,
        nhead,
        nhid_factor,
        nlayers,
        n_features,
        child_model="mlp",
        tree_depth=3,
        dropout=0.0,
        y_encoder_layer=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        all_layers_same_init=False,
        efficient_eval_masking=True,
        decoder_type="output_attention",
        predicted_hidden_layer_size=None,
        decoder_embed_dim=2048,
        classification_task=True,
        decoder_hidden_layers=1,
        decoder_hidden_size=None,
        predicted_hidden_layers=1,
        weight_embedding_rank=None,
        y_encoder=None,
        low_rank_weights=False,
        tabpfn_zero_weights=True,
        decoder_activation="relu",
        predicted_activation="relu",
    ):
        super().__init__()
        self.child_model = child_model
        self.classification_task = classification_task
        self.tree_depth = tree_depth

        # decoder activation = "relu" is legacy behavior
        nhid = emsize * nhid_factor

        # mothernet has batch_first=False, unlike all the other models.
        def encoder_layer_creator():
            return TransformerEncoderLayer(
                emsize,
                nhead,
                nhid,
                dropout,
                activation=activation,
                pre_norm=pre_norm,
                recompute_attn=recompute_attn,
                batch_first=False,
            )

        self.transformer_encoder = TransformerEncoderSimple(
            encoder_layer_creator, nlayers
        )

        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        if wandb.run:
            wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.decoder_activation = decoder_activation
        self.emsize = emsize
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.y_encoder = y_encoder_layer
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_out = n_out
        self.nhid = nhid
        self.decoder_type = decoder_type
        decoder_hidden_size = decoder_hidden_size or nhid
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.predicted_activation = predicted_activation

        if self.child_model == "mlp":
            self.decoder = MLPModelDecoder(
                emsize=emsize,
                hidden_size=decoder_hidden_size or nhid,
                n_out=n_out,
                decoder_type=decoder_type,
                predicted_hidden_layer_size=predicted_hidden_layer_size,
                embed_dim=decoder_embed_dim,
                decoder_hidden_layers=decoder_hidden_layers,
                nhead=nhead,
                predicted_hidden_layers=predicted_hidden_layers,
                weight_embedding_rank=weight_embedding_rank,
                low_rank_weights=low_rank_weights,
                decoder_activation=decoder_activation,
                in_size=n_features,
            )
        elif self.child_model == "gradtree":
            self.decoder = GradTreeDecoder(
                emsize=emsize,
                hidden_size=decoder_hidden_size or nhid,
                n_out=n_out,
                decoder_type=decoder_type,
                embed_dim=decoder_embed_dim,
                decoder_hidden_layers=decoder_hidden_layers,
                in_size=n_features,
                tree_depth=tree_depth,
            )
        else:
            raise ValueError(f"Unknown child_model type: {self.child_model}")

        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))

        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = (
                    layer.self_attn
                    if isinstance(layer.self_attn, nn.ModuleList)
                    else [layer.self_attn]
                )
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def inner_forward(self, train_x):
        return self.transformer_encoder(train_x)
