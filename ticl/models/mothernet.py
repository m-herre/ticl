import torch, wandb
import torch.nn as nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F

from ticl.models.encoders import OneHotAndLinear
from ticl.models.decoders import MLPModelDecoder, GradTreeDecoder
from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.models.encoders import Linear

from ticl.utils import SeqBN, get_init_method

import numpy as np


# ————— helpers —————
def one_hot_argmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    # hardmax → one-hot along 'dim'
    idx = logits.argmax(dim=dim, keepdim=True)
    oh = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
    return oh


def st(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    # straight-through: forward=hard, backward=soft
    return soft - (soft - hard.detach())


def entmax15(**kwargs):
    return


class ModelPredictor(nn.Module):

    def tree_forward(self, x_test, I_logits, T, L, n_actual_features):
        """
        Paper-exact GradTree pass with ST operators (Algorithm 1) for Ensembles (GRANDE).

        Args:
            x_test:   (n_test, batch, n_features)
            I_logits: (batch, n_estimators, n_nodes, n_features)
            T:        (batch, n_estimators, n_nodes) — scalar threshold per node
            L:        (batch, n_estimators, n_leaves, n_out)
            n_actual_features: Number of real features (before padding)

        Returns:
            y_pred: (n_test, batch, n_out)
        """
        dtype = x_test.dtype

        # 1) Feature selection: entmax then ST → hard one-hot with soft gradients
        I_logits_masked = I_logits.clone()
        # mask on last dim (features)
        I_logits_masked[..., n_actual_features:] = -float("inf")

        # I_soft/hard: (batch, est, nodes, n_feat)
        I_soft = F.softmax(I_logits_masked, dim=-1)
        I_hard = one_hot_argmax(I_logits_masked, dim=-1).to(dtype)
        I = st(I_hard, I_soft)

        # 2) Split probability per node
        # T is (batch, est, nodes) — scalar threshold, no feature-dim dot product needed

        # <I,x> x_test is (test, batch, feat), I is (batch, est, nodes, feat)
        # Result needs to be (test, batch, est, nodes)
        s2_sum = torch.einsum("tbn,bein->tbei", x_test, I)

        # Broadcast T to (1, batch, est, nodes)
        s_soft = (F.softsign(T.unsqueeze(0) - s2_sum) + 1) / 2

        s_hard = torch.round(s_soft)  # (t, b, e, n_nodes)
        s = st(s_hard, s_soft)

        if torch.isnan(s).any():
            print("⚠️ NaN detected in split probabilities! Check T / x_proj magnitudes.")

        # 3) Path probabilities over levels
        # self.decoder.internal_node_index_list shape: (n_leaves, depth)
        # s shape: (t, b, e, n_nodes)
        # We gather specific nodes for path calculation
        s_selected = s[
            ..., self.decoder.internal_node_index_list
        ]  # (t, b, e, n_leaves, depth)

        path_ids = self.decoder.path_identifier_list  # (n_leaves, depth)

        # Product over depth (last dim)
        p_leaves = torch.prod(
            ((1 - path_ids) * s_selected + path_ids * (1.0 - s_selected)),
            dim=-1,
        )  # (t, b, e, l)

        # 4) Expected leaf aggregation
        # L: (b, e, leaves, out)
        y_hat_estimators = torch.einsum(
            "tbel,belo->tbeo", p_leaves, L
        )  # (t, b, e, n_out)

        # 5) Ensemble Aggregation (Average over estimators)
        y_hat = y_hat_estimators.mean(dim=2)  # (t, b, n_out)

        if torch.isnan(y_hat).any():
            print("❌ NaN detected in final output logits!")
            print(
                "=> Check for exploding values in t_proj/x_proj or invalid entmax output."
            )

        return y_hat

    def forward(self, src, single_eval_pos=None):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src
        info, x, y = src

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

            # Get actual feature count from info dict (before padding)
            n_actual_features = (
                info.get("num_features_used", x.shape[-1])
                if info is not None
                else x.shape[-1]
            )

            # 🔧 Use soft differentiable tree inference
            h = self.tree_forward(x_test, I_logits, T, L, n_actual_features)

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
        n_estimators=10,
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
                n_estimators=n_estimators,
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
