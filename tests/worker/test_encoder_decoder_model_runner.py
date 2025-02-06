# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import List

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner

BATCH_SIZES = [1, 4, 16, 64, 256]


def _create_model_runner(model: str, *args,
                         **kwargs) -> EncoderDecoderModelRunner:
    engine_args = EngineArgs(model, *args, **kwargs)
    engine_config = engine_args.create_engine_config()
    model_runner = EncoderDecoderModelRunner(
        vllm_config=engine_config,
        is_driver_worker=True,
    )
    return model_runner


@pytest.mark.parametrize("batch_size", list(range(1, 257)))
@pytest.mark.parametrize("multiple_seqs_per_seq_group", [True, False])
def test_prepare_decode_cuda_graph(batch_size, multiple_seqs_per_seq_group):
    """
    Tests that for encoder-decoder models with CUDA Graph capture and replay
    enabled, the tensors used during the decode phase are correctly padded
    for varying input batch sizes.
    """
    model_runner = _create_model_runner(
        "facebook/bart-base",
        seed=0,
        dtype="float16",
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
        enforce_eager=False,
    )
    block_tables = {
        0: [1],
        1: [3]
    } if multiple_seqs_per_seq_group else {
        0: [1]
    }
    seq_lens: List[int] = []
    encoder_seq_lens: List[int] = []
    seq_group_metadata_list: List[SequenceGroupMetadata] = []

    cross_block_table = [2]
    expanded_batch_size = 0
    for i in range(batch_size):
        # make sure all tokens fit into one block
        seq_len = i % (model_runner.block_size - 1) + 1
        seq_data = SequenceData.from_seqs(range(seq_len))
        encoder_seq_len = (i + 1) % (model_runner.block_size - 1) + 1
        encoder_seq_data = SequenceData.from_seqs(range(encoder_seq_len))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={
                0: seq_data,
                1: seq_data
            } if multiple_seqs_per_seq_group else {0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
            encoder_seq_data=encoder_seq_data,
            cross_block_table=cross_block_table,
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_lens.extend(
            [seq_len for _ in range(len(seq_group_metadata.seq_data))])
        encoder_seq_lens.extend(
            [encoder_seq_len for _ in range(len(seq_group_metadata.seq_data))])
        expanded_batch_size = expanded_batch_size + len(
            seq_group_metadata.seq_data)
        seq_group_metadata_list.append(seq_group_metadata)

    model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    input_tokens = model_input.input_tokens
    input_positions = model_input.input_positions
    attn_metadata = model_input.attn_metadata
    return_seq_lens = model_input.seq_lens
    slot_mapping = attn_metadata.slot_mapping
    encoder_input_tokens = model_input.encoder_input_tokens
    encoder_input_positions = model_input.encoder_input_positions
    cross_slot_mapping = attn_metadata.cross_slot_mapping

    # With CUDA Graph capture and replay enabled, the decoder and encoder
    # input sequences will be padded. Create the expected padded tensors
    # accordingly.
    graph_batch_size = model_runner.vllm_config.pad_for_cudagraph(
        expanded_batch_size)
    cuda_graph_pad_size = graph_batch_size - expanded_batch_size
    padded_seq_lens = seq_lens + list(itertools.repeat(1, cuda_graph_pad_size))
    padded_encoder_seq_lens = encoder_seq_lens + list(
        itertools.repeat(1, cuda_graph_pad_size))

    assert return_seq_lens == padded_seq_lens
    assert len(slot_mapping) == len(input_tokens)
    assert len(cross_slot_mapping) == len(encoder_input_tokens)

    # Verify attention metadata
    device = model_runner.device
    assert attn_metadata.num_prefills == 0
    assert attn_metadata.num_decode_tokens > 0
    assert torch.equal(
        attn_metadata.seq_lens_tensor,
        torch.tensor(padded_seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.seq_lens == padded_seq_lens
    assert attn_metadata.max_prefill_seq_len == 0
    assert attn_metadata.max_decode_seq_len == max(seq_lens)
    # - Encoder attention metadata
    assert attn_metadata.encoder_seq_lens == padded_encoder_seq_lens
    assert torch.equal(
        attn_metadata.encoder_seq_lens_tensor,
        torch.tensor(padded_encoder_seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.max_encoder_seq_len == max(padded_encoder_seq_lens)
    assert attn_metadata.num_encoder_tokens == sum(padded_encoder_seq_lens)

    # Verify block tables are correct for prompts
    # - Decoder self-attention. Pad the block tables as expected.
    flattened_block_tables = [
        block_table for _ in range(len(seq_group_metadata_list))
        for block_table in block_tables.values()
    ]
    flattened_block_tables.extend([[] for _ in range(cuda_graph_pad_size)])
    expected = make_tensor_with_pad(
        flattened_block_tables,
        max_len=64,
        pad=0,
        dtype=torch.int32,
        device=model_runner.device,
    )
    assert torch.equal(
        attn_metadata.block_tables,
        expected,
    )
    # - Encoder/decoder cross-attention. Pad the cross-attention block tables
    # as expected.
    expected = [
        cross_block_table for seq_group_metadata in seq_group_metadata_list
        for _ in range(len(seq_group_metadata.seq_data))
    ]
    expected.extend([[] for _ in range(cuda_graph_pad_size)])
    expected = make_tensor_with_pad(
        expected,
        max_len=64,
        pad=0,
        dtype=torch.int32,
        device=model_runner.device,
    )
    assert torch.equal(
        attn_metadata.cross_block_tables,
        expected,
    )

    # Model runner's CUDAGraph setting should be propagated to attention
    # metadata.
    assert attn_metadata.use_cuda_graph is True

    # Verify the lengths of input tokens & positions
    # - Decoder
    assert len(input_tokens) == len(padded_seq_lens)
    assert len(input_positions) == len(padded_seq_lens)
    # -- An indirect check that model_input.input_tokens
    #    and model_input.input_positions are correct -
    #    by design of the test, the input tokens are
    #    equal to the input position values, so if
    #    the model_input data structure has the correct
    #    values then these two should be equal
    assert torch.equal(
        input_tokens,
        input_positions,
    )
    # - Encoder
    assert len(encoder_input_tokens) == 0
    assert len(encoder_input_tokens) == 0
    # -- An indirect check that model_input.encoder_input_tokens
    #    and model_input.encoder_input_positions are correct -
    #    by design of the test, the input tokens are
    #    equal to the input position values, so if
    #    the model_input data structure has the correct
    #    values then these two should be equal
    assert torch.equal(
        encoder_input_tokens,
        encoder_input_positions,
    )
