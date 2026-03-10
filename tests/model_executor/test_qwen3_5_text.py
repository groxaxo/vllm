# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM


def test_qwen35_text_mrope_positions_are_1d():
    model = object.__new__(Qwen3_5ForCausalLM)

    positions, delta = model.get_mrope_input_positions([11, 12, 13], [])

    assert delta == 0
    assert positions.shape == (3, 3)
    expected = torch.arange(3, dtype=torch.long)
    assert torch.equal(positions[0], expected)
    assert torch.equal(positions[1], expected)
    assert torch.equal(positions[2], expected)


def test_qwen35_text_mapper_handles_vl_prefixes():
    mapped_names = Qwen3_5ForCausalLM.hf_to_vllm_mapper.apply_list(
        [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.visual.blocks.0.weight",
        ]
    )

    assert mapped_names == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.0.weight",
    ]
