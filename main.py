import asyncio
import json
import logging
import os

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("claude_code_llm")

app = FastAPI()


class BrowserFrameSerializer(FrameSerializer):
    """Wire protocol between pipecat and the browser.

    Incoming binary  → InputAudioRawFrame  (16 kHz, mono, PCM-16)
    Incoming text    → InputTransportMessageFrame (JSON dict)
    Outgoing audio   → raw bytes  (WAV when add_wav_header=True)
    Outgoing message → JSON string
    """

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        if isinstance(data, str):
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = data
            return InputTransportMessageFrame(message=msg)
        return None


class InputModeTracker:
    """Shared flag: True = mic input (TTS on), False = text input (TTS off)."""

    def __init__(self):
        self.tts_enabled = True


class TextInputProcessor(FrameProcessor):
    """Converts {"type":"text_input","text":"..."} browser messages into pipeline frames."""

    def __init__(self, mode: InputModeTracker):
        super().__init__()
        self._mode = mode

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict) and msg.get("type") == "text_input":
                text = msg.get("text", "").strip()
                if text:
                    self._mode.tts_enabled = False  # text turn — suppress TTS
                    await self.push_frame(TranscriptionFrame(text=text, user_id="", timestamp=""))
                    await self.push_frame(UserStoppedSpeakingFrame())
                return  # consume; don't forward the raw message
        await self.push_frame(frame, direction)


class TTSGate(FrameProcessor):
    """Drops TextFrames going to TTS when the turn came from text input.
    Also re-enables TTS after each response so the next mic turn works normally."""

    def __init__(self, mode: InputModeTracker):
        super().__init__()
        self._mode = mode

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and not self._mode.tts_enabled:
            return  # drop — TTS should not speak for text-mode turns
        if isinstance(frame, LLMFullResponseEndFrame):
            self._mode.tts_enabled = True  # re-arm for next turn
        await self.push_frame(frame, direction)


class ClaudeCodeLLMService(FrameProcessor):
    """Runs Claude Code via `sbx exec <sandbox> claude -p ...` and streams the response.

    Consumes LLMContextFrame, emits LLMFullResponseStartFrame / TextFrame(s) /
    LLMFullResponseEndFrame — the same contract as any pipecat LLM service.
    """

    def __init__(
        self,
        sandbox_name: str = "claude-yolo",
        permission_mode: str = "bypassPermissions",
        allowed_tools: str | None = None,
    ):
        super().__init__()
        self._sandbox_name = sandbox_name
        self._permission_mode = permission_mode
        self._allowed_tools = allowed_tools

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: list[dict]) -> str:
        """Flatten the conversation history into a single prompt string."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Content may be a list of blocks (e.g. vision format)
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, role.capitalize())
            parts.append(f"{label}: {content}")
        return "\n".join(parts)

    def _build_cmd(self, prompt: str) -> list[str]:
        cmd = [
            "sbx", "exec", self._sandbox_name,
            "claude", "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", self._permission_mode,
        ]
        if self._allowed_tools:
            cmd += ["--allowedTools", self._allowed_tools]
        return cmd

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        messages = frame.context.messages
        prompt = self._build_prompt(messages)
        cmd = self._build_cmd(prompt)

        logger.info(">>> Command: %s %s", " ".join(cmd), prompt)

        await self.push_frame(LLMFullResponseStartFrame())

        response_chunks: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async for line in proc.stdout:
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    # Not JSON — forward as plain text (shouldn't happen with stream-json)
                    await self.push_frame(TextFrame(text=raw))
                    continue

                event_type = event.get("type")
                if event_type == "assistant":
                    # Extract text blocks from the assistant message content
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            response_chunks.append(block["text"])
                            await self.push_frame(TextFrame(text=block["text"]))
                elif event_type == "result" and event.get("is_error"):
                    error_msg = event.get("result", "unknown error")
                    await self.push_frame(TextFrame(text=f"[Error: {error_msg}]"))

            await proc.wait()

            if proc.returncode != 0:
                stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
                if stderr:
                    logger.error("<<< stderr: %s", stderr)
                    await self.push_frame(TextFrame(text=f"[Error: {stderr}]"))

        except Exception as e:
            logger.exception("<<< exception running Claude Code")
            await self.push_frame(TextFrame(text=f"[Claude Code error: {e}]"))

        logger.info("<<< response:\n%s", "".join(response_chunks))
        await self.push_frame(LLMFullResponseEndFrame())


class UserTranscriptSender(FrameProcessor):
    """Sends user transcripts to the browser so they appear in the chat UI."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(
                OutputTransportMessageFrame(
                    message={"type": "transcript", "role": "user", "text": frame.text}
                )
            )
        await self.push_frame(frame, direction)


class BotTextSender(FrameProcessor):
    """Streams LLM text chunks to the browser so the response appears while audio plays."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            await self.push_frame(OutputTransportMessageFrame(message={"type": "bot_start"}))
        elif isinstance(frame, TextFrame) and frame.text:
            await self.push_frame(
                OutputTransportMessageFrame(message={"type": "bot_text", "text": frame.text})
            )
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(OutputTransportMessageFrame(message={"type": "bot_end"}))
        await self.push_frame(frame, direction)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    task: PipelineTask | None = None

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
            add_wav_header=True,
            serializer=BrowserFrameSerializer(),
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        if task is not None:
            await task.cancel()

    llm = ClaudeCodeLLMService(
        sandbox_name=os.getenv("CLAUDE_CODE_SANDBOX", "claude-yolo"),
        permission_mode=os.getenv("CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"),
        allowed_tools=os.getenv("CLAUDE_CODE_ALLOWED_TOOLS") or None,
    )

    stt = ElevenLabsRealtimeSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        commit_strategy=CommitStrategy.MANUAL,
    )

    async with aiohttp.ClientSession() as aiohttp_session:
        tts = ElevenLabsHttpTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            aiohttp_session=aiohttp_session,
        )

        context = LLMContext(
            messages=[
                # {
                #     "role": "system",
                #     "content": "You are a helpful, friendly assistant. Keep responses concise.",
                # }
            ]
        )
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        mode = InputModeTracker()

        pipeline = Pipeline(
            [
                transport.input(),
                TextInputProcessor(mode),
                stt,
                UserTranscriptSender(),
                context_aggregator.user(),
                llm,
                BotTextSender(),
                TTSGate(mode),
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, enable_rtvi=False)
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
