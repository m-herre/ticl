import torch, wandb
import torch.nn as nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F

from ticl.models.encoders import OneHotAndLinear
from ticl.models.decoders import MLPModelDecoder, GradTreeDecoder
from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.models.encoders import Linear

from ticl.utils import SeqBN, get_init_method


# ————— helpers —————
def one_hot_argmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    # hardmax → one-hot along 'dim'
    idx = logits.argmax(dim=dim, keepdim=True)
    oh = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
    return oh


def st(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    # straight-through: forward=hard, backward=soft
    return hard + (soft - soft.detach())


# 🔧 Placeholder for entmax15 (will replace later with proper implementation)
def entmax15(inputs, dim=-1, eps=1e-9):
    """
    Computes the 1.5-entmax transformation (Peters et al., 2019).
    Equivalent to softmax but yields sparse probabilities.

    Args:
        inputs: Tensor of shape (..., n)
        dim: Dimension to apply entmax along
        eps: Numerical stability constant

    Returns:
        Tensor of same shape, where entries are >= 0 and sum to 1 along `dim`.
    """

    # Step 1: sort inputs along dim
    X = inputs.transpose(dim, -1)
    d = X.size(-1)
    X_sorted, _ = torch.sort(X, descending=True)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype).view(1, -1)

    # Step 2: compute running means and squared means
    X_cumsum = torch.cumsum(X_sorted, dim=-1)
    X_sq_cumsum = torch.cumsum(X_sorted**2, dim=-1)

    # Step 3: Compute τ candidates (thresholds)
    mean_sq = X_sq_cumsum / rho
    mean = X_cumsum / rho
    S = (1.0 - mean_sq + mean**2).clamp_min(0)
    τ_candidates = mean - torch.sqrt(S)

    # Step 4: find support size (the largest k such that τ < x_k)
    support = (τ_candidates < X_sorted).type(X_sorted.dtype)
    support_size = support.sum(dim=-1, keepdim=True)

    # Step 5: τ* = τ at support size
    τ_star = τ_candidates.gather(-1, (support_size - 1).long().clamp(min=0))
    τ_star = τ_star.transpose(dim, -1)

    # Step 6: compute final probabilities
    output = ((inputs - τ_star).clamp_min(0)) ** 2
    output /= output.sum(dim=dim, keepdim=True).clamp_min(eps)

    return output


class ModelPredictor(nn.Module):

    def tree_forward_soft(self, x_test, I_logits, T, L):
        """
        Paper-exact GradTree pass with ST operators (Algorithm 1).

        Args:
            x_test:   (n_test, batch, n_features)
            I_logits: (batch, n_nodes, n_features)
            T:        (batch, n_nodes, n_features)
            L:        (batch, n_leaves, n_out)

        Returns:
            y_pred: (n_test, batch, n_out)
        """
        n_test, batch, n_features = x_test.shape
        n_nodes = I_logits.shape[1]
        depth = self.tree_depth
        n_leaves = 2**depth
        device = x_test.device
        dtype = x_test.dtype

        def stat(name, t):
            n_nan = torch.isnan(t).sum().item()
            # print(
            #     f"[DEBUG] {name}: shape={tuple(t.shape)}, "
            #     f"mean={t.mean().item():.4e}, std={t.std().item():.4e}, "
            #     f"min={t.min().item():.4e}, max={t.max().item():.4e}, "
            #     f"NaNs={n_nan}",
            #     flush=True,
            # )

        # 1) Feature selection: entmax then ST → hard one-hot with soft gradients
        I_soft = entmax15(I_logits, dim=-1)  # (b, n_nodes, n_feat)
        I_hard = one_hot_argmax(I_logits, dim=-1).to(dtype)  # (b, n_nodes, n_feat)
        I = st(
            I_hard, I_soft
        )  # ST entmax (paper)  [Alg.1, line 2-3], :contentReference[oaicite:3]{index=3}
        stat("I_logits", I_logits)
        stat("I_soft (entmax)", I_soft)
        stat("I_hard", I_hard)
        stat("I (ST)", I)

        # 2) Split probability per node (left prob “s” in the paper)
        #    s = sigmoid( <I,T> - <I,x> )  (Eq. 6), then ST rounding to {0,1} (Alg.1 line 11)
        #    <I,T> is batch×nodes, <I,x> is test×batch×nodes
        t_proj = (I * T).sum(-1)  # (b, n_nodes)        <I,T>
        x_proj = torch.einsum("tbf,bnf->tbn", x_test, I)  # (t, b, n_nodes)      <I,x>
        s_soft = torch.sigmoid(
            t_proj.unsqueeze(0) - x_proj
        )  # (t, b, n_nodes)      Eq. (6) as in Alg.1 line 10, :contentReference[oaicite:4]{index=4}
        s_hard = torch.round(s_soft)  # (t, b, n_nodes)
        s = st(
            s_hard, s_soft
        )  # ST on split (paper)  [Alg.1, line 11], :contentReference[oaicite:5]{index=5}
        # Note: s is the LEFT probability. RIGHT probability is (1 - s).
        # (No clamping here to follow the paper exactly.)
        stat("T", T)
        stat("t_proj", t_proj)
        stat("x_proj", x_proj)
        diff = t_proj.unsqueeze(0) - x_proj
        stat("diff (t_proj - x_proj)", diff)
        stat("s_soft (sigmoid)", s_soft)
        stat("s_hard", s_hard)
        stat("s (ST split prob)", s)

        if torch.isnan(s).any():
            print("⚠️ NaN detected in split probabilities! Check T / x_proj magnitudes.")

        # 3) Path probabilities over levels (paper’s L(x|l,·) via p(l,j) bits; Alg.1 line 12)
        p_nodes = torch.ones(
            (n_test, batch, 1), device=device, dtype=dtype
        )  # (t, b, 1)
        node_offset = 0
        for level in range(depth):
            n_level = 2**level
            s_lvl = s[..., node_offset : node_offset + n_level]  # (t, b, n_level)
            # Expand parent probs for left/right children
            p_nodes = p_nodes.repeat_interleave(2, dim=-1)  # (t, b, 2*n_level)
            # Left = s, Right = 1 - s   (matches Eq. 3’s ((1-p)*s + p*(1-s)) with p∈{0,1})
            left = s_lvl
            right = 1.0 - s_lvl
            children = torch.stack([left, right], dim=-1).reshape(
                n_test, batch, -1
            )  # interleave
            p_nodes = p_nodes * children
            node_offset += n_level
            stat(f"p_nodes (level {level})", p_nodes)

        p_leaves = p_nodes  # (t, b, n_leaves)

        # 4) Expected leaf aggregation (Alg.1 line 14–16)
        y_hat = torch.einsum("tbl,blo->tbo", p_leaves, L)  # (t, b, n_out)
        # The paper applies softmax at the end to get class probabilities (Alg.1 line 16).
        # Return logits or probabilities depending on your training loop.
        # If you want paper-exact behavior (probabilities), uncomment the next line:
        # y_hat = F.softmax(y_hat, dim=-1)                                     # :contentReference[oaicite:6]{index=6}

        stat("L (leaf logits)", L)
        stat("y_hat (output logits)", y_hat)

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

            # 🔧 Use soft differentiable tree inference
            h = self.tree_forward_soft(x_test, I_logits, T, L)

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
