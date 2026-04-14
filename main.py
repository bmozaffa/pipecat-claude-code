import asyncio
import json
import logging
import os
import shlex

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


class TTSToggle:
    """Shared flag controlled by the browser toggle. True = speak responses aloud."""

    def __init__(self):
        self.enabled = True


class TextInputProcessor(FrameProcessor):
    """Converts browser messages into pipeline frames.

    {"type":"text_input","text":"..."} → TranscriptionFrame + UserStoppedSpeakingFrame
    {"type":"set_tts","enabled":bool}  → updates TTSToggle
    """

    def __init__(self, tts_toggle: TTSToggle):
        super().__init__()
        self._tts_toggle = tts_toggle

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict):
                if msg.get("type") == "text_input":
                    text = msg.get("text", "").strip()
                    if text:
                        await self.push_frame(TranscriptionFrame(text=text, user_id="", timestamp=""))
                        await self.push_frame(UserStoppedSpeakingFrame())
                    return  # consume; don't forward the raw message
                if msg.get("type") == "set_tts":
                    self._tts_toggle.enabled = bool(msg.get("enabled", True))
                    logger.info("TTS toggled: %s", self._tts_toggle.enabled)
                    return  # consume
        await self.push_frame(frame, direction)


class TTSGate(FrameProcessor):
    """Drops TextFrames going to TTS when the speak-response toggle is off."""

    def __init__(self, tts_toggle: TTSToggle):
        super().__init__()
        self._tts_toggle = tts_toggle

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and not self._tts_toggle.enabled:
            return  # drop — user toggled speaking off
        await self.push_frame(frame, direction)


class ClaudeCodeLLMService(FrameProcessor):
    """Runs Claude Code and streams the response.

    Supports two execution modes:
      - "sbx"  (default): runs via ``sbx exec <sandbox> claude -p ...``
      - "ssh": runs via SSH.  Two sub-variants:
          * Without ``ssh_script``: ``ssh <user>@<host> -- claude -p <prompt> ...``
          * With ``ssh_script``: ``echo <prompt> | ssh <user>@<host> -- bash -lc <script>``
            (prompt is fed via stdin; the remote script owns the claude invocation)

    Consumes LLMContextFrame, emits LLMFullResponseStartFrame / TextFrame(s) /
    LLMFullResponseEndFrame — the same contract as any pipecat LLM service.
    """

    def __init__(
        self,
        execution_mode: str = "sbx",
        sandbox_name: str = "claude-yolo",
        ssh_host: str | None = None,
        ssh_user: str | None = None,
        ssh_script: str | None = None,
        claude_path: str = "claude",
        permission_mode: str = "bypassPermissions",
        allowed_tools: str | None = None,
    ):
        super().__init__()
        if execution_mode not in ("sbx", "ssh"):
            raise ValueError(f"execution_mode must be 'sbx' or 'ssh', got {execution_mode!r}")
        self._execution_mode = execution_mode
        self._sandbox_name = sandbox_name
        self._ssh_host = ssh_host
        self._ssh_user = ssh_user
        self._ssh_script = ssh_script
        self._claude_path = claude_path
        self._permission_mode = permission_mode
        self._allowed_tools = allowed_tools
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_latest_user_message(self, messages: list[dict]) -> str:
        """Extract the most recent user message from the context."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
                return content
        return ""

    def _build_claude_args(self, prompt: str) -> list[str]:
        """Return the claude sub-command arguments (shared between sbx and ssh)."""
        args = [self._claude_path, "-p", prompt]
        if self._session_id:
            args += ["--resume", self._session_id]
        args += [
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", self._permission_mode,
        ]
        if self._allowed_tools:
            args += ["--allowedTools", self._allowed_tools]
        return args

    def _build_cmd(self, prompt: str) -> list[str]:
        if self._execution_mode == "ssh":
            if self._ssh_script:
                # Headless mode: prompt is fed via stdin; the remote script owns claude.
                return ["ssh", f"{self._ssh_user}@{self._ssh_host}", "--", "bash", "-lc", self._ssh_script]
            # Direct mode: pass all claude args over SSH.
            remote_cmd = shlex.join(self._build_claude_args(prompt))
            return ["ssh", f"{self._ssh_user}@{self._ssh_host}", "--", remote_cmd]
        # Default: sbx exec
        return ["sbx", "exec", self._sandbox_name] + self._build_claude_args(prompt)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        messages = frame.context.messages
        prompt = self._get_latest_user_message(messages)
        cmd = self._build_cmd(prompt)
        use_stdin = self._execution_mode == "ssh" and bool(self._ssh_script)

        logger.info(">>> Command: %s", " ".join(cmd))

        await self.push_frame(LLMFullResponseStartFrame())

        response_chunks: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if use_stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if use_stdin:
                proc.stdin.write(prompt.encode())
                await proc.stdin.drain()
                proc.stdin.close()

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
                elif event_type == "result":
                    session_id = event.get("session_id")
                    if session_id:
                        self._session_id = session_id
                        logger.info("<<< session_id: %s", session_id)
                    if event.get("is_error"):
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
        execution_mode=os.getenv("CLAUDE_CODE_EXECUTION_MODE", "sbx"),
        sandbox_name=os.getenv("CLAUDE_CODE_SANDBOX", "claude-yolo"),
        ssh_host=os.getenv("CLAUDE_CODE_SSH_HOST") or None,
        ssh_user=os.getenv("CLAUDE_CODE_SSH_USER") or None,
        ssh_script=os.getenv("CLAUDE_CODE_SSH_SCRIPT") or None,
        claude_path=os.getenv("CLAUDE_CODE_CLAUDE_PATH", "claude"),
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
            settings=ElevenLabsHttpTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            ),
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

        tts_toggle = TTSToggle()

        pipeline = Pipeline(
            [
                transport.input(),
                TextInputProcessor(tts_toggle),
                stt,
                UserTranscriptSender(),
                context_aggregator.user(),
                llm,
                BotTextSender(),
                TTSGate(tts_toggle),
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
