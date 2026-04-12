import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
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
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

load_dotenv()

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


class TextInputProcessor(FrameProcessor):
    """Converts {"type":"text_input","text":"..."} browser messages into pipeline frames."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict) and msg.get("type") == "text_input":
                text = msg.get("text", "").strip()
                if text:
                    await self.push_frame(TranscriptionFrame(text=text, user_id="", timestamp=""))
                    await self.push_frame(UserStoppedSpeakingFrame())
                return  # consume; don't forward the raw message
        await self.push_frame(frame, direction)


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

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", "none"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    stt = ElevenLabsRealtimeSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        commit_strategy=CommitStrategy.VAD,
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, friendly assistant. Keep responses concise.",
            }
        ]
    )
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            TextInputProcessor(),
            stt,
            UserTranscriptSender(),
            context_aggregator.user(),
            llm,
            BotTextSender(),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
