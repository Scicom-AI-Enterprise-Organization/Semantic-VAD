import io

import numpy as np
import soundfile as sf

from semantic_vad.audio import turn_to_row
from semantic_vad.build import write_parquet
from semantic_vad.parsers import normalize_words
from semantic_vad.schema import TurnConfig
from semantic_vad.turns import build_turns
from tests.fixtures import MULTILINGUAL_DURATION, MULTILINGUAL_WORDS


def _make_rows(sr=16000, n=3):
    words = normalize_words(MULTILINGUAL_WORDS)
    turn = build_turns(words, MULTILINGUAL_DURATION, TurnConfig(mode="single"))[0]
    array = np.random.default_rng(0).standard_normal(int(7.2 * sr)).astype(np.float32) * 0.1
    return [
        turn_to_row(turn, array, sr, row_id=f"en__x__turn_{i:03d}", language="en")
        for i in range(n)
    ]


def test_write_parquet_schema_and_audio_roundtrip(tmp_path):
    import json

    import pyarrow.parquet as pq

    out = tmp_path / "en.parquet"
    n = write_parquet(iter(_make_rows(n=3)), str(out), sampling_rate=16000, batch_size=2)
    assert n == 3

    table = pq.read_table(str(out))
    assert table.num_rows == 3
    assert table.column_names == [
        "id", "audio", "language", "duration", "silence_spans", "words", "messages",
    ]
    # audio is a {bytes, path} struct.
    assert {f.name for f in table.schema.field("audio").type} == {"bytes", "path"}

    # HF feature metadata is embedded and marks audio as an Audio feature.
    meta = json.loads(table.schema.metadata[b"huggingface"])
    feats = meta["info"]["features"]
    assert feats["audio"]["_type"] == "Audio"
    assert feats["audio"]["sampling_rate"] == 16000
    assert feats["silence_spans"][0]["start"]["_type"] == "Value"

    # Audio bytes decode back to a real 16 kHz waveform (as eot-bench reads them).
    first = table.to_pylist()[0]
    arr, sr = sf.read(io.BytesIO(first["audio"]["bytes"]), dtype="float32")
    assert sr == 16000
    assert abs(len(arr) / sr - first["duration"]) < 0.05
    # Last silence span is the EOT and lies within the clip.
    assert first["silence_spans"][-1]["end"] <= first["duration"] + 1e-6


def test_written_parquet_loads_as_audio_dataset(tmp_path):
    # datasets should recognize the Audio feature from the embedded metadata,
    # and reading with decode=False must not require torch/torchcodec.
    from datasets import Audio, load_dataset

    out = tmp_path / "en.parquet"
    write_parquet(iter(_make_rows(n=2)), str(out), sampling_rate=16000)
    ds = load_dataset("parquet", data_files=str(out), split="train")
    assert isinstance(ds.features["audio"], Audio)
    ds = ds.cast_column("audio", Audio(decode=False))
    arr, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]), dtype="float32")
    assert sr == 16000 and len(arr) > 0
