import time
import warnings

import torch, wandb
import torch.nn as nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F

from ticl.models.encoders import OneHotAndLinear
from ticl.models.decoders import MLPModelDecoder, GradTreeDecoder, GrandeDecoder
from ticl.models.grande_core import (
    flatten_grande_estimator_outputs,
    grande_forward,
    grande_forward_compiled,
    pairwise_cosine_off_diag,
)
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
    return soft - (soft - hard).detach()


def entmax15(**kwargs):
    return


class ModelPredictor(nn.Module):
    @staticmethod
    def _retain_gradients(tensors):
        if tensors is None:
            return
        if torch.is_tensor(tensors):
            if tensors.requires_grad:
                tensors.retain_grad()
            return
        if isinstance(tensors, dict):
            for value in tensors.values():
                ModelPredictor._retain_gradients(value)
            return
        if isinstance(tensors, (list, tuple)):
            for value in tensors:
                ModelPredictor._retain_gradients(value)

    def _grande_profile_active(self):
        return self.child_model == "grande" and getattr(self, "grande_profile", False)

    def _sync_profile_device(self, device):
        if self._grande_profile_active() and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _start_grande_timer(self, device):
        if not self._grande_profile_active():
            return None
        self._sync_profile_device(device)
        return time.perf_counter()

    def _stop_grande_timer(self, name, start, device):
        if start is None:
            return
        self._sync_profile_device(device)
        self.grande_profile_totals[name] = (
            self.grande_profile_totals.get(name, 0.0)
            + (time.perf_counter() - start)
        )

    def reset_grande_profile(self):
        self.grande_profile_totals = {}
        self.grande_profile_steps = 0

    def get_grande_profile_stats(self, reset=False):
        if not self._grande_profile_active() or self.grande_profile_steps == 0:
            return {}
        stats = {
            name: value / self.grande_profile_steps
            for name, value in self.grande_profile_totals.items()
        }
        stats["grande_total_s"] = sum(stats.values())
        stats["grande_profile_steps"] = self.grande_profile_steps
        if reset:
            self.reset_grande_profile()
        return stats

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

        # 1) Feature selection: softmax then ST → hard one-hot with soft gradients
        mask = torch.arange(I_logits.shape[-1], device=I_logits.device) >= n_actual_features
        I_logits_masked = I_logits.masked_fill(mask, -float("inf"))

        # I_soft/hard: (batch, est, nodes, n_feat)
        I_soft = F.softmax(I_logits_masked, dim=-1)
        I_hard = one_hot_argmax(I_logits_masked, dim=-1).to(dtype)
        I = st(I_hard, I_soft)

        # 2) Split probability per node
        # T is (batch, est, nodes) — scalar threshold
        # <I,x> selects the chosen feature value per node (gradient flows through I via ST)
        s2_sum = torch.einsum("tbn,bein->tbei", x_test, I)

        # Broadcast T to (1, batch, est, nodes)
        s_soft = (F.softsign(T.unsqueeze(0) - s2_sum) + 1) / 2

        s_hard = torch.round(s_soft)  # (t, b, e, n_nodes)
        s = st(s_hard, s_soft)

        # 3) Path probabilities over levels
        s_selected = s[
            ..., self.decoder.internal_node_index_list
        ]  # (t, b, e, n_leaves, depth)

        path_ids = self.decoder.path_identifier_list  # (n_leaves, depth)

        p_leaves = torch.prod(
            ((1 - path_ids) * s_selected + path_ids * (1.0 - s_selected)),
            dim=-1,
        )  # (t, b, e, l)

        # 4) Leaf aggregation → ensemble mean
        y_hat_estimators = torch.einsum(
            "tbel,belo->tbeo", p_leaves, L
        )  # (t, b, e, n_out)

        return y_hat_estimators.mean(dim=2)  # (t, b, n_out)

    @staticmethod
    def _compute_grande_diversity_aux_losses(
        *,
        split_values,
        split_index_logits,
        estimator_weights,
        leaf_classes,
    ):
        combined_vectors = flatten_grande_estimator_outputs(
            split_values=split_values,
            split_index_logits=split_index_logits,
            estimator_weights=estimator_weights,
            leaf_classes=leaf_classes,
        )
        off_diag = pairwise_cosine_off_diag(combined_vectors)
        zero = combined_vectors.new_zeros(())
        if off_diag is None:
            return {
                "grande_diversity_loss": zero,
                "grande_diversity_cosine_mean": zero,
                "grande_diversity_positive_cosine_mean": zero,
            }
        positive_mean = off_diag.clamp_min(0).mean()
        return {
            "grande_diversity_loss": positive_mean,
            "grande_diversity_cosine_mean": off_diag.mean(),
            "grande_diversity_positive_cosine_mean": positive_mean,
        }

    def forward(
        self,
        src,
        single_eval_pos=None,
        return_debug=False,
        return_aux=False,
        grande_context_seed=None,
        advance_split_temperature=True,
        grande_use_training_schedule=None,
    ):
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
        layer_outputs = None
        if return_debug:
            output, layer_outputs = self.inner_forward(
                enc_train,
                return_layer_outputs=True,
            )
        else:
            output = self.inner_forward(enc_train)
        aux_losses = {}

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

        elif self.child_model == "grande":
            # Fix train/eval coordinate mismatch (Point 5): use stored
            # random state when no explicit seed given, so
            # features_by_estimator is deterministic during training
            # just as it is at inference.
            if grande_context_seed is None:
                grande_context_seed = self.decoder.grande_random_state
            num_features_used = (
                info.get("num_features_used", x.shape[-1]) if info is not None else x.shape[-1]
            )
            context_start = self._start_grande_timer(x.device)
            context = self.decoder.build_context(
                batch_size=x.shape[1],
                num_features_used=num_features_used,
                device=x.device,
                seed=grande_context_seed,
            )
            self._stop_grande_timer("grande_context_s", context_start, x.device)
            decoder_out = self.decoder(
                output,
                y[:single_eval_pos],
                x[:single_eval_pos],
                context,
                seed=grande_context_seed,
                return_profile=self._grande_profile_active(),
                return_debug=return_debug,
                advance_split_temperature=advance_split_temperature,
                use_training_schedule=grande_use_training_schedule,
            )
            decoder_debug = None
            if self._grande_profile_active() and return_debug:
                (
                    split_values,
                    split_index_logits,
                    estimator_weights,
                    leaf_classes,
                    decoder_extra,
                ) = decoder_out
                decoder_timings = decoder_extra["timings"]
                decoder_debug = {
                    key: value for key, value in decoder_extra.items() if key != "timings"
                }
                for name, value in decoder_timings.items():
                    self.grande_profile_totals[name] = (
                        self.grande_profile_totals.get(name, 0.0) + value
                    )
            elif self._grande_profile_active():
                (
                    split_values,
                    split_index_logits,
                    estimator_weights,
                    leaf_classes,
                    decoder_timings,
                ) = decoder_out
                for name, value in decoder_timings.items():
                    self.grande_profile_totals[name] = (
                        self.grande_profile_totals.get(name, 0.0) + value
                    )
            elif return_debug:
                (
                    split_values,
                    split_index_logits,
                    estimator_weights,
                    leaf_classes,
                    decoder_debug,
                ) = decoder_out
            else:
                split_values, split_index_logits, estimator_weights, leaf_classes = decoder_out
            forward_start = self._start_grande_timer(x.device)
            grande_out = self._grande_forward(
                x=x[single_eval_pos:],
                split_values=split_values,
                split_index_logits=split_index_logits,
                estimator_weights=estimator_weights,
                leaf_classes=leaf_classes,
                features_by_estimator=context["features_by_estimator"],
                feature_mask=context["feature_mask"],
                path_identifier_list=self.decoder.path_identifier_list_float,
                internal_node_index_list=self.decoder.internal_node_index_list,
                training=self.training,
                dropout=self.decoder.grande_dropout,
                missing_values=self.decoder.missing_values,
                straight_through=True,
                return_debug=return_debug,
            )
            self._stop_grande_timer("grande_forward_s", forward_start, x.device)
            if self._grande_profile_active():
                self.grande_profile_steps += 1
            if return_aux and getattr(self, "grande_diversity_loss_weight", 0.0) > 0.0:
                aux_losses = self._compute_grande_diversity_aux_losses(
                    split_values=split_values,
                    split_index_logits=split_index_logits,
                    estimator_weights=estimator_weights,
                    leaf_classes=leaf_classes,
                )
            if return_debug:
                h, grande_debug = grande_out
            else:
                h = grande_out

        else:
            raise ValueError(f"Unknown child_model type: {self.child_model}")

        if h.isnan().all():
            print("NAN")
            raise ValueError("NAN")
        if return_debug:
            debug = {
                "encoder_output": enc_train,
                "transformer_layer_outputs": layer_outputs or [],
                "transformer_output": output,
                "single_eval_pos": single_eval_pos,
            }
            if self.child_model == "grande":
                debug.update(
                    {
                        "split_values": split_values,
                        "split_index_logits": split_index_logits,
                        "estimator_weights": estimator_weights,
                        "leaf_classes": leaf_classes,
                        "features_by_estimator": context["features_by_estimator"],
                        "feature_mask": context["feature_mask"],
                        "num_features_used": int(num_features_used),
                        "tree_depth": self.decoder.tree_depth,
                    }
                )
                if decoder_debug is not None:
                    debug.update(decoder_debug)
                debug.update(grande_debug)
                self._retain_gradients(
                    [
                        debug["encoder_output"],
                        debug["transformer_output"],
                        debug["transformer_layer_outputs"],
                        debug["split_values"],
                        debug["split_index_logits"],
                        debug["estimator_weights"],
                        debug["leaf_classes"],
                    ]
                )
            if return_aux:
                return h, debug, aux_losses
            return h, debug
        if return_aux:
            return h, aux_losses
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
        selected_variables=16,
        data_subset_fraction=1.0,
        bootstrap=False,
        grande_dropout=0.0,
        missing_values=True,
        grande_random_state=42,
        grande_decoder_variant="baseline",
        grande_output_init="default",
        grande_split_temperature_start=1.0,
        grande_split_temperature_end=1.0,
        grande_split_temperature_anneal_steps=0,
        grande_compile=False,
        grande_profile=False,
        grande_diagnostics=False,
        grande_diagnostics_gradients=False,
        grande_diagnostics_level="scalars_small_hists",
        grande_diagnostics_seed=0,
        grande_diagnostics_hist_max_points=2048,
        grande_diversity_loss_weight=0.0,
    ):
        super().__init__()
        self.child_model = child_model
        self.classification_task = classification_task
        self.tree_depth = tree_depth
        self.grande_profile = grande_profile
        self.grande_diagnostics = grande_diagnostics
        self.grande_diagnostics_gradients = grande_diagnostics_gradients
        self.grande_diagnostics_level = grande_diagnostics_level
        self.grande_diagnostics_seed = grande_diagnostics_seed
        self.grande_diagnostics_hist_max_points = grande_diagnostics_hist_max_points
        self.grande_diversity_loss_weight = float(grande_diversity_loss_weight)
        self._grande_forward = grande_forward
        self.reset_grande_profile()
        if self.grande_diversity_loss_weight < 0.0:
            raise ValueError("grande_diversity_loss_weight must be non-negative")
        if self.grande_diversity_loss_weight > 0.0 and self.child_model != "grande":
            raise ValueError(
                "grande_diversity_loss_weight is only supported for child_model='grande'"
            )
        if (
            self.grande_diversity_loss_weight > 0.0
            and grande_decoder_variant != "factorized_stats"
        ):
            raise ValueError(
                "grande_diversity_loss_weight requires grande_decoder_variant='factorized_stats'"
            )

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
            wandb.log({"backbone_size": backbone_size}, step=0)
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
        elif self.child_model == "grande":
            self.decoder = GrandeDecoder(
                emsize=emsize,
                hidden_size=decoder_hidden_size or nhid,
                n_out=n_out,
                decoder_type=decoder_type,
                embed_dim=decoder_embed_dim,
                decoder_hidden_layers=decoder_hidden_layers,
                nhead=nhead,
                decoder_activation=decoder_activation,
                in_size=n_features,
                tree_depth=tree_depth,
                n_estimators=n_estimators,
                selected_variables=selected_variables,
                data_subset_fraction=data_subset_fraction,
                bootstrap=bootstrap,
                grande_dropout=grande_dropout,
                missing_values=missing_values,
                grande_random_state=grande_random_state,
                grande_decoder_variant=grande_decoder_variant,
                grande_output_init=grande_output_init,
                grande_split_temperature_start=grande_split_temperature_start,
                grande_split_temperature_end=grande_split_temperature_end,
                grande_split_temperature_anneal_steps=grande_split_temperature_anneal_steps,
            )
            if grande_compile:
                if not hasattr(torch, "compile"):
                    warnings.warn(
                        "grande_compile was requested, but torch.compile is unavailable; "
                        "falling back to eager GRANDE forward.",
                        RuntimeWarning,
                    )
                else:
                    try:
                        self._grande_forward = torch.compile(
                            grande_forward_compiled, dynamic=True
                        )
                    except Exception as exc:
                        warnings.warn(
                            "grande_compile was requested, but torch.compile setup failed; "
                            f"falling back to eager GRANDE forward. Original error: {exc}",
                            RuntimeWarning,
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

    def inner_forward(self, train_x, return_layer_outputs=False):
        return self.transformer_encoder(
            train_x,
            return_layer_outputs=return_layer_outputs,
        )
