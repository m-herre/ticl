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
        Paper-exact GradTree pass with ST operators (Algorithm 1).

        Args:
            x_test:   (n_test, batch, n_features)
            I_logits: (batch, n_nodes, n_features)
            T:        (batch, n_nodes, n_features)
            L:        (batch, n_leaves, n_out)
            n_actual_features: Number of real features (before padding)

        Returns:
            y_pred: (n_test, batch, n_out)
        """
        dtype = x_test.dtype

        # 1) Feature selection: entmax then ST → hard one-hot with soft gradients
        # print(f"n_actual_features in tree_forward: {n_actual_features}")
        I_logits_masked = I_logits.clone()
        I_logits_masked[:, :, n_actual_features:] = -float("inf")

        I_soft = F.softmax(I_logits_masked, dim=-1)  # (b, n_nodes, n_feat)
        I_hard = one_hot_argmax(I_logits_masked, dim=-1).to(
            dtype
        )  # (b, n_nodes, n_feat)
        I = st(
            I_hard, I_soft
        )  # ST entmax (paper)  [Alg.1, line 2-3], :contentReference[oaicite:3]{index=3}

        # 2) Split probability per node (left prob “s” in the paper)
        #    s = sigmoid( <I,T> - <I,x> )  (Eq. 6), then ST rounding to {0,1} (Alg.1 line 11)
        #    <I,T> is batch×nodes, <I,x> is test×batch×nodes

        s1_sum = torch.einsum(
            "bin,bin->bi", T, I
        )  # b batch; t n_test; n n_features; i n_nodes
        s2_sum = torch.einsum("tbn,bin->tbi", x_test, I)

        s_soft = (F.softsign(s1_sum - s2_sum) + 1) / 2

        s_hard = torch.round(s_soft)  # (t, b, n_nodes)
        s = st(
            s_hard, s_soft
        )  # ST on split (paper)  [Alg.1, line 11], :contentReference[oaicite:5]{index=5}
        # Note: s is the LEFT probability. RIGHT probability is (1 - s).
        # (No clamping here to follow the paper exactly.)

        if torch.isnan(s).any():
            print("⚠️ NaN detected in split probabilities! Check T / x_proj magnitudes.")

        # 3) Path probabilities over levels (paper’s L(x|l,·) via p(l,j) bits; Alg.1 line 12)
        s_selected = s[..., self.decoder.internal_node_index_list]
        p_leaves = torch.prod(
            (
                (1 - self.decoder.path_identifier_list) * s_selected
                + self.decoder.path_identifier_list * (1.0 - s_selected)
            ),
            dim=3,
        )  # (t, b, l)

        # 4) Expected leaf aggregation (Alg.1 line 14–16)
        y_hat = torch.einsum("tbl,blo->tbo", p_leaves, L)  # (t, b, n_out)

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
