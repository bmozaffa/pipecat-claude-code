# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv` for dependency management (Python 3.11).

```bash
uv sync                  # Install dependencies from uv.lock
uv add <package>         # Add a new dependency
uv run python main.py    # Run the app
uv run python -m pytest  # Run tests (if added)
```

## Project Overview

A [pipecat-ai](https://github.com/pipecat-ai/pipecat) application scaffolded with the `elevenlabs` extra. Pipecat is a framework for building real-time voice and multimodal AI pipelines. The core abstraction is a **pipeline** of **frames** flowing through **processors** (services, transports, aggregators).

Key pipecat concepts:
- **Frames**: typed data units (audio, text, LLM messages, control signals) that flow through the pipeline
- **Processors**: async components that consume and emit frames (e.g., `ElevenLabsTTSService`, `OpenAILLMService`, `DailyTransport`)
- **Pipeline**: connects processors; typically `Transport → STT → LLM → TTS → Transport`
- **Runner**: drives the pipeline event loop (`PipelineRunner`, `PipelineTask`)

The current `main.py` is a placeholder — real pipecat apps construct and run a pipeline there.
