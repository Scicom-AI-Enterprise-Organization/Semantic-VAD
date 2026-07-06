"""Architecture smoke test -- no checkpoint download, no GPU required.

Builds a tiny random-weight Qwen2Audio config (same classes as the real
7B model, just small), runs a forward+backward pass through
`Qwen2AudioEoTClassifier`, and asserts shapes/gradients look right. This is
meant to catch merge/pooling bugs cheaply; it says nothing about model quality.
"""

import sys

import torch
from transformers import Qwen2AudioConfig, Qwen2AudioEncoderConfig, Qwen2AudioModel
from transformers.models.qwen2 import Qwen2Config

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from semvad.modeling import EoTHead, EoTHeadConfig, Qwen2AudioEoTClassifier  # noqa: E402


def build_tiny_config() -> Qwen2AudioConfig:
    num_mel_bins = 8
    max_source_positions = 8  # -> expected mel length = max_source_positions * 1 * 2 = 16
    audio_config = Qwen2AudioEncoderConfig(
        num_mel_bins=num_mel_bins,
        d_model=32,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        encoder_layers=2,
        max_source_positions=max_source_positions,
    )
    text_config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return Qwen2AudioConfig(audio_config=audio_config, text_config=text_config, audio_token_index=10)


def main():
    torch.manual_seed(0)
    config = build_tiny_config()

    backbone = Qwen2AudioModel(config)
    backbone.audio_tower.requires_grad_(False)
    head_config = EoTHeadConfig(head_hidden_size=16, dropout=0.0)
    model = Qwen2AudioEoTClassifier(backbone, EoTHead(config.text_config.hidden_size, head_config))
    model.train()

    batch_size = 2
    mel_len = config.audio_config.max_source_positions * 1 * 2  # conv1 stride=1, conv2 stride=2
    # audio_feat_lengths=(mel_len-1)//2+1=8 ; audio_output_lengths=(8-2)//2+1=4 -> 4 audio placeholder tokens
    num_audio_tokens = 4

    input_features = torch.randn(batch_size, config.audio_config.num_mel_bins, mel_len)
    feature_attention_mask = torch.ones(batch_size, mel_len, dtype=torch.long)

    # sequence: [BOS_TEXT, <AUDIO> x4, TAIL_TEXT x2, PAD?] right-padded
    audio_token_id = config.audio_token_index
    seq_len = 1 + num_audio_tokens + 2
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        input_ids[b, 0] = 1
        input_ids[b, 1 : 1 + num_audio_tokens] = audio_token_id
        input_ids[b, 1 + num_audio_tokens :] = torch.randint(11, 63, (2,))

    # make row 1 shorter (right-padded) to exercise the pooling index math
    attention_mask[1, -1] = 0
    input_ids[1, -1] = 0

    labels = torch.tensor([1.0, 0.0])

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        labels=labels,
    )

    assert output.logits.shape == (batch_size,), output.logits.shape
    assert output.p_eot.shape == (batch_size,)
    assert torch.all((output.p_eot >= 0) & (output.p_eot <= 1))
    assert output.loss is not None and output.loss.item() == output.loss.item()  # not NaN

    output.loss.backward()
    head_grad_norm = sum(p.grad.abs().sum().item() for p in model.head.parameters() if p.grad is not None)
    assert head_grad_norm > 0, "head received no gradient"

    audio_tower_grads = [p.grad for p in model.backbone.audio_tower.parameters()]
    assert all(g is None for g in audio_tower_grads), "audio tower was frozen -- should get no gradient"

    n_total = model.count_parameters()
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"OK. total_params={n_total:,} head_params={n_head:,} loss={output.loss.item():.4f} p_eot={output.p_eot.tolist()}")


if __name__ == "__main__":
    main()
