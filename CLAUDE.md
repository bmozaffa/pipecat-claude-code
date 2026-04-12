# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv` for dependency management (Python 3.11).

```bash
uv sync                  # Install / update dependencies from uv.lock
uv add <package>         # Add a new dependency
uv run python main.py    # Run the app (or: uv run uvicorn main:app --reload)
```

## Running the App

1. Copy `.env.example` → `.env` and fill in your API keys.
2. Start the server:
   ```bash
   uv run python main.py
   ```
3. Open `http://localhost:7860` in a browser.

## Environment Variables

| Variable | Purpose |
|---|---|
| `ELEVENLABS_API_KEY` | ElevenLabs API key (STT + TTS) |
| `ELEVENLABS_VOICE_ID` | TTS voice (default: Rachel) |
| `OPENAI_API_KEY` | Key for the LLM endpoint |
| `OPENAI_BASE_URL` | Base URL for any OpenAI-compatible endpoint (leave blank for OpenAI) |
| `OPENAI_MODEL` | Model name (default: `gpt-4o-mini`) |

## Architecture

`main.py` is a FastAPI app that runs a pipecat voice pipeline over WebSocket.

**Transport**: `FastAPIWebsocketTransport` (port 7860, `/ws`). A custom `BrowserFrameSerializer` maps pipecat frames to the wire protocol:
- Incoming binary → `InputAudioRawFrame` (16 kHz mono PCM-16)
- Incoming JSON text → `InputTransportMessageFrame`
- Outgoing audio → WAV bytes (binary frame)
- Outgoing messages → JSON text frame

**Pipeline** (linear, per WebSocket connection):
```
transport.input()
  → TextInputProcessor       # {"type":"text_input"} → TranscriptionFrame + UserStoppedSpeakingFrame
  → ElevenLabsRealtimeSTTService (CommitStrategy.VAD)  # audio → TranscriptionFrame
  → UserTranscriptSender     # copies TranscriptionFrame → OutputTransportMessageFrame for chat UI
  → context_aggregator.user()
  → OpenAILLMService         # configurable base_url for any OpenAI-compatible endpoint
  → BotTextSender            # streams LLMFullResponseStart/TextFrame/End → OutputTransportMessageFrame
  → ElevenLabsTTSService     # text → audio (WAV chunks sent to browser)
  → transport.output()
  → context_aggregator.assistant()
```

**Browser → Server JSON messages**:
- `{"type": "text_input", "text": "..."}` — text-mode input

**Server → Browser JSON messages**:
- `{"type": "transcript", "role": "user", "text": "..."}` — STT result
- `{"type": "bot_start"}` — bot begins responding
- `{"type": "bot_text", "text": "..."}` — streaming text chunk
- `{"type": "bot_end"}` — bot finished responding

## Key pipecat Concepts

- **Frames** — typed data units flowing through the pipeline (audio, text, control signals)
- **FrameProcessor** — base class for all pipeline stages; override `process_frame(frame, direction)`, call `super()` first to handle system frames, then `push_frame(frame, direction)` to forward
- **LLMContext / LLMContextAggregatorPair** — manages conversation history; `context_aggregator.user()` collects transcripts, `context_aggregator.assistant()` captures LLM replies
- **CommitStrategy.VAD** — ElevenLabs handles its own voice-activity detection for STT segmentation (no separate Silero VAD needed)
