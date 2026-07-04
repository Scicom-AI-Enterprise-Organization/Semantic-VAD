# INTEGRATION.md — serving our EOT model into LiveKit Agents

How to plug the model trained on this dataset (**Whisper encoder → adapter → Qwen3 → EOT
head**, see the architecture section in `README.md`) into a LiveKit voice agent as a
drop-in replacement for LiveKit's built-in turn detector.

**Decision:** we host the model ourselves on **GPU** and reuse LiveKit's **stock cloud
transport** — LiveKit streams the user's audio to our server over a websocket and we return
an end-of-turn probability. No changes to the LiveKit client are required; we only implement
the server end of their protocol.

## Background: LiveKit has two EOT interfaces

| | Classic plugin `livekit-plugins-turn-detector` | **New** `livekit.agents.inference.eot` |
|---|---|---|
| Input | transcript text (`ChatContext`, last 6 turns / 128 tokens) | **streaming audio** (16 kHz PCM) |
| Entry | `EOUModelBase.predict_end_of_turn(chat_ctx) -> float` | `TurnDetector(...).stream()` |
| Model | ONNX transformer over text | v1 (cloud websocket) / v1-mini (local ctypes) |

Our model is audio-native, so we target the **new** `inference/eot` interface. The classic
text plugin is irrelevant to us.

## How the new interface is structured

- **`TurnDetector`** (`detector.py`) — public class, re-exported as
  `from livekit.agents.inference import TurnDetector`. Picks a transport by version:
  - `version="v1"` → `_CloudTransport` — websocket to a remote inference server.
  - `version="v1-mini"` → `_LocalTransport` — in-process ctypes model (`<500 MB` CPU).
  - Sticky one-way **fallback**: on cloud error/timeout it degrades to the local mini and
    never returns to cloud for that session.
- **`_BaseStreamingTurnDetectorStream`** (`base.py`) — the per-session engine (audio
  ingress, resampling to 16 kHz, request/future plumbing, fallback). We reuse it as-is.
- **`_CloudTransport`** (`transports.py`) — streams audio frames to `{base_url}/eot` and
  resolves predictions. **This is the client we serve.**

Because we reuse `version="v1"`, everything above stays stock. We implement only the
websocket server it talks to.

## Wiring the agent to our GPU

Set these before constructing the agent, and use `TurnDetector(version="v1")`:

```bash
export LIVEKIT_INFERENCE_URL="https://our-gpu-host:PORT"   # http(s) → ws(s) auto
export LIVEKIT_API_KEY="..."                               # used for the Bearer token
export LIVEKIT_API_SECRET="..."
```

`_CloudTransport` connects to **`{base_url}/eot`** (the `http`→`ws` scheme swap is
automatic) and attaches `Authorization: Bearer <token>` (minted from the API key/secret)
plus LiveKit inference headers. Our server must accept (or ignore) that header — see the
auth note at the end.

## The wire protocol our server implements

Binary **protobuf** frames from `livekit.protocol.agent_pb.agent_inference`. All audio is
**16 kHz, mono, PCM S16LE** (`SessionSettings.sample_rate=16000`,
`encoding=AUDIO_ENCODING_PCM_S16LE`).

### Client (LiveKit) → our GPU — `ClientMessage` (oneof `message`)

| field | when | payload |
|---|---|---|
| `session_create` | once, on connect | `SessionSettings{ sample_rate, encoding }` |
| `input_audio` | continuously, per frame | `InputAudio{ audio: PCM bytes, num_samples, created_at }` |
| `inference_start` | after VAD detects ≥ 200 ms silence | `InferenceStart{ request_id }` |
| `session_flush` | turn boundary | `SessionFlush{}` → **reset our per-session buffer** |
| `session_close` | teardown | `SessionClose{}` |

(Each `ClientMessage` also carries a top-level `created_at` timestamp, used for latency
telemetry — echo/ignore as needed.)

### Our GPU → client — `ServerMessage` (oneof `message`, plus `request_id`)

| field | when | payload |
|---|---|---|
| `session_created` | once, after `session_create` | `default_thresholds` (per-lang map) + `default_threshold` (+ optional backchannel maps) — **our calibrated thresholds; the client adopts them** |
| `eot_prediction` | per `inference_start` | `EotPrediction{ probability, backchannel_probability?, inference_stats }`, `request_id` = the matching request |
| `session_closed` / `inference_started` / `inference_stopped` | acks | optional, carry timestamps for latency |
| `error` | failure | `{ message, code }` |

## Serving semantics that matter for the model

- **We accumulate the audio buffer server-side.** Audio is streamed frame-by-frame; the
  whole turn lives on our GPU. `session_flush` clears it. This is exactly the **per-turn
  Whisper+Qwen3 window** — the client sends everything, we decide the context window
  (cap ≤ 30 s to match Whisper's positional range and the training distribution).
- **Request/response is keyed by `request_id`.** Run the forward pass on `inference_start`,
  tag the reply with the same id. The client ignores stale replies and always supersedes
  with the latest request, so we don't need to cancel in-flight work — just answer the
  newest id.
- **~1 s latency budget.** If we don't reply within `DEFAULT_PREDICTION_TIMEOUT = 1.0 s`,
  the client abandons the request and **falls back to the local mini** (sticky). So
  Whisper-encode + Qwen3-prefill + EOT-head over the accumulated turn must finish under 1 s
  on our GPU — feasible with a small Qwen3 (0.6B/1.7B) + base/small Whisper encoder.
- **We return a raw probability.** Thresholding is client-side against the per-language
  `unlikely_threshold` we ship in `session_created` (LiveKit's own values sit ~0.20–0.40).
  Calibrate these on our eval set and send them; don't bake the decision into the
  probability.
- **VAD gating is the client's job.** We only receive `inference_start` at genuine pause
  candidates (≥ 200 ms silence). We never poll mid-word — matching how this dataset's
  `silence_spans` mark the decision points.
- **Backchannel is optional.** Omit `backchannel_probability` if we don't model it (the mini
  doesn't either).

## Server sketch

```python
# pseudo-code — an aiohttp/websockets server exposing /eot
async def eot_ws(request):
    ws = await accept(request)                      # validate Bearer here if desired
    buf = TurnAudioBuffer(max_seconds=30, sample_rate=16000)
    model = WhisperQwen3EOT.load(...)               # on GPU

    async for frame in ws:                          # binary protobuf
        msg = ClientMessage.parse(frame.data)
        kind = msg.WhichOneof("message")

        if kind == "session_create":
            await ws.send(ServerMessage(session_created=SessionCreated(
                default_thresholds=OUR_LANG_THRESHOLDS,   # calibrated per language
                default_threshold=OUR_LANG_THRESHOLDS["en"],
            )).SerializeToString())

        elif kind == "input_audio":
            buf.append(msg.input_audio.audio)         # PCM S16LE @ 16 kHz

        elif kind == "inference_start":
            pcm = buf.read()                          # whole turn so far
            prob = await run_on_gpu(model, pcm)       # must return < 1 s
            await ws.send(ServerMessage(
                request_id=msg.inference_start.request_id,
                eot_prediction=EotPrediction(probability=prob),
            ).SerializeToString())

        elif kind == "session_flush":
            buf.clear()                               # turn boundary → new turn

        elif kind == "session_close":
            break
```

## Serving with vLLM

The `WhisperQwen3EOT` / `run_on_gpu(...)` step in the sketch above is a **vLLM** instance
serving a small custom audio model. The end-of-turn score comes straight from **token
logprobs** — no bespoke classification head.

### Model choice

- Build a **small** model: Whisper (base/small) encoder → adapter → **dense Qwen3**
  (0.6B / 1.7B) → logprob read. Sized to fit the ~1 s budget before LiveKit falls back.
- **Not Qwen3-Omni-30B-A3B.** vLLM serves it, but it's an omni *chat* MoE — orders of
  magnitude too heavy for a turn detector. It's only a reference that Qwen3+audio wiring is
  a solved problem to crib from.
- Use **vLLM ≥ 0.17** — that's where the audio-cache / mixed-modality handling got fixed
  (relevant because we stream a growing turn buffer).

### Getting the model into vLLM

Qwen2-Audio's encoder **is** a Whisper-large-v3 encoder, so our architecture ≈ Qwen2-Audio
with a Qwen3 backbone — i.e. a "Qwen3-Audio". Two ways to add it, preferred first:

1. **Out-of-tree plugin (no fork).** A `vllm.general_plugins` entry point that calls
   `ModelRegistry.register_model("Qwen3AudioForConditionalGeneration", YourClass)`, with the
   class implementing `SupportsMultiModal` and registering its processor on
   `MULTIMODAL_REGISTRY`. Ships as our own package — no rebasing vLLM each release.
2. **Thin fork (fallback).** If the multimodal registration hooks fight us (they're a less
   stable API than plain-LM registration), fork and adapt `vllm/model_executor/models/qwen2_audio.py` directly.

Either way, the delta from `qwen2_audio` is small — **reuse** the audio tower, projector,
and multimodal processor (mel featurization, `<|audio_bos|><|AUDIO|><|audio_eos|>`
placeholder expansion, embedding-merge); **change** only:

- swap `Qwen2Model` → `Qwen3Model` (already in vLLM `qwen3.py`) — Qwen3 adds **QK-norm**,
  drops **QKV bias**, uses explicit `head_dim`;
- a new config class composing the audio-encoder config + a Qwen3 text config;
- `load_weights` name-mapping (Qwen3 has `q_norm` / `k_norm` params Qwen2 lacks).

### EOT via logprobs (no classification head)

At the decision point, do a single probe step and read the marker-token probability — this
is exactly how LiveKit's *text* detector works (its score is the EOU-token probability):

```python
out = llm.generate(prompt_embeds=embeds, sampling_params=SamplingParams(
    max_tokens=1, logprobs=k))                    # k large enough to include both markers
lp = out[0].outputs[0].logprobs[0]                # {token_id: logprob}
p_eot = softmax([lp[EOT_ID], lp[CONTINUE_ID]])[0] # calibrated 2-way probability
```

- **Token choice:** reuse an existing turn-end token (e.g. chat `<|im_end|>`) or add a
  dedicated `<eot>` — if you add one, train its embedding during fine-tune.
- **Calibrate as a 2-way contrast** (`{eot, continue}` softmax), not a bare single-token
  logprob (which competes with the whole vocab and isn't a calibrated turn-probability).
- **Top-k gotcha:** vLLM returns top-`k` logprobs; if a marker token isn't in the top-`k` at
  a mid-turn hold you get nothing for it — bump `k` or rely on the 2-token contrast.
- Return this **raw probability** to LiveKit; the `unlikely_threshold`s live client-side
  (ship them in `session_created`, calibrated on this scale).

### Two gotchas at the fine-tune → serve handoff

- **Placeholder count must match.** The processor expands `<|AUDIO|>` into *N* tokens; *N*
  must equal the adapter's per-frame output, and the marker tokens / chat template must be
  identical between training and serving. Mismatch = silent garbage.
- **Prefix-cache vs. bidirectional encoder.** If Whisper encodes the growing turn buffer
  bidirectionally, past audio-token embeddings change each `inference_start`, so vLLM's
  automatic prefix cache misses on the audio span. Fine at a coarse VAD-gated cadence over a
  `< 30 s` turn — just don't expect free KV reuse across requests within a turn.

## Auth note

The stock client always attaches a LiveKit Bearer token to the WS handshake. Two options:

1. **Zero-change drop-in** — have our server validate or simply ignore that header. Then
   `TurnDetector(version="v1")` + the env vars above is a genuine drop-in; nothing on the
   agent changes.
2. **Own auth / own protocol** — swap in a trivial custom transport implementing
   `_StreamingTurnDetectionTransport` (`run` / `run_inference` / `push_frame` / `flush` /
   `attach` / `detach`) that speaks whatever protocol we prefer. More code, full control.

We go with (1): honor the header, reuse the stock cloud path.

## How the dataset feeds the model

Each row of [`Scicom-intl/semantic-vad-eot`](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot)
is one turn: `words` (content + order) supervise the semantic branch; `silence_spans` (EOT
positionally last) are the `hold`/`eot` targets and mark the decision points the client's
VAD gate fires on at serving time. See `README.md` for the full architecture and training
rationale.

## Source

LiveKit `agents@main`, `livekit-agents/livekit/agents/inference/eot/`:
[`base.py`](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/inference/eot/base.py) ·
[`detector.py`](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/inference/eot/detector.py) ·
[`transports.py`](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/inference/eot/transports.py) ·
[`languages.py`](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/inference/eot/languages.py)
