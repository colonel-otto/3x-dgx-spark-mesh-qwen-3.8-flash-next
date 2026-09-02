from vllm.model_executor.models.interfaces import SupportsMRoPE
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextHyperConnection,
    Qwen3NextModel,
    Qwen4ExpTextRMSNorm,
    QwenNextMixtureOfExperts,
    _all_gather_hidden_and_residual,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig

from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3NextMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen3NextConfig = model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.hidden_size = config.hidden_size
        self.has_hyper_connection = getattr(config, "hc_count", 0) > 1
        self.hc_count = getattr(config, "hc_count", 4) if self.has_hyper_connection else 1
        self.hc_dim = self.hc_count * config.hidden_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.fc_embedding = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.fc_hidden",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hc_dim if self.has_hyper_connection else self.hidden_size
        )

        if self.has_hyper_connection:
            self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(
                self.hc_dim, group_size=self.hidden_size, eps=config.rms_norm_eps
            )
            self.hyper_connection_mixer = Qwen3NextHyperConnection(
                config, prefix=maybe_prefix(prefix, "hyper_connection_mixer"), use_combine=False
            )
            self.norm = None
        else:
            self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.hyper_connection_mixer = None
            self.norm = Qwen4ExpTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds, _ = self.fc_embedding(inputs_embeds)

            if self.has_hyper_connection:
                num_tokens = hidden_states.shape[0]
                if hidden_states.shape[-1] == self.hidden_size:
                    hidden_states = hidden_states.unsqueeze(1).expand(-1, self.hc_count, -1)
                else:
                    hidden_states = hidden_states.view(num_tokens, self.hc_count, self.hidden_size)
                hidden_normed = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                    num_tokens, self.hc_count, self.hidden_size
                )
                hidden_proj, _ = self.fc_hidden(hidden_normed)
                hidden_states = (inputs_embeds.unsqueeze(-2) + hidden_proj).flatten(-2)
            else:
                hidden_normed = self.pre_fc_norm_hidden(hidden_states)
                hidden_proj, _ = self.fc_hidden(hidden_normed)
                hidden_states = inputs_embeds + hidden_proj

            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.has_hyper_connection:
            hidden_states = self.hyper_connection_mixer(hidden_states)
        else:
            if mtp_layer.use_attn_reduce_scatter_for_moe:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    positions.shape[-1],
                    self.config.hidden_size,
                )
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _remap(w):
            for name, tensor in w:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model."):]
                elif name.startswith("model.visual.") or name.startswith("visual."):
                    continue
                if (
                    ".ple." in name
                    or name.endswith(".ple")
                    or ".indexer." in name
                ):
                    continue
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "")
                    yield name, tensor
                else:
                    yield name, tensor

        weights = maybe_fuse_shared_experts(
            _remap(weights),
            n_routed_experts=self.config.num_experts,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_prefixes=["ple", "indexer"],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3NextMTP(nn.Module, QwenNextMixtureOfExperts, SupportsMRoPE):
    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3NextMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        shared_weight_names = ["embed_tokens", "lm_head"]

        def remap_weight_names(weights):
            fc_hidden = None
            fc_embedding = None
            for name, weight in weights:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model."):]
                elif name.startswith("model.visual.") or name.startswith("visual."):
                    continue
                if (
                    ".ple." in name
                    or name.endswith(".ple")
                    or ".indexer." in name
                ):
                    continue
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                    yield name, weight
                elif any(key in name for key in shared_weight_names):
                    yield name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_prefixes=["ple", "indexer"],
        )
        return loader.load_weights(remap_weight_names(weights))
