"""
Benchmark script for measuring TTFB with HTTP-based TTS services.

Compares TTFB across HTTP-based TTS providers:
- Inworld (HTTP)
- ElevenLabs (HTTP)
- Cartesia (HTTP)

This simulates an LLM returning tokens one at a time, with the TTS service
aggregating them into complete sentences before sending to the provider.
"""

import asyncio
import os
import struct
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

load_dotenv(override=True)


class SimulatedLLMProcessor(FrameProcessor):
    """Simulates an LLM returning tokens one at a time."""

    def __init__(self, text: str, token_delay_ms: float = 50, **kwargs):
        """
        Args:
            text: The full text to emit as tokens
            token_delay_ms: Delay between tokens in milliseconds (simulates LLM generation speed)
        """
        super().__init__(**kwargs)
        self._text = text
        self._token_delay_s = token_delay_ms / 1000.0
        self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            # Trigger token emission after start
            if not self._started:
                self._started = True
                asyncio.create_task(self._emit_tokens())
        else:
            await self.push_frame(frame, direction)

    async def _emit_tokens(self):
        """Emit tokens one at a time with delays."""
        await asyncio.sleep(0.1)  # Small initial delay

        # Signal start of LLM response
        await self.push_frame(LLMFullResponseStartFrame())

        # Split text into words (tokens)
        words = self._text.split()

        for i, word in enumerate(words):
            # Add space before word (except first)
            if i > 0:
                token = " " + word
            else:
                token = word

            logger.debug(f"Emitting token: '{token}'")
            await self.push_frame(TextFrame(text=token))
            await asyncio.sleep(self._token_delay_s)

        # Signal end of LLM response
        await self.push_frame(LLMFullResponseEndFrame())

        # Small delay then end
        await asyncio.sleep(0.5)
        await self.push_frame(EndFrame())


class TTFBCollector(FrameProcessor):
    """Collects TTFB metrics, word timestamps, and audio frames for analysis."""

    def __init__(
        self,
        service_name: str = "TTS",
        save_audio: bool = True,
        output_dir: str = "benchmark_audio",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.service_name = service_name
        self.save_audio = save_audio
        self.output_dir = output_dir
        self.ttfb_values = []
        self.audio_frame_count = 0
        self.total_audio_bytes = 0
        self.first_audio_time = None
        self.first_chunk_bytes = None
        self.sample_rate = None
        self.all_audio_chunks = []  # Accumulate all audio
        self.start_time = None
        self.last_audio_time = None
        self._done_event = asyncio.Event()
        self._check_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Start a task to check for audio completion
            self._check_task = asyncio.create_task(self._check_audio_done())
        elif isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, TTFBMetricsData):
                    # Filter out spurious near-zero TTFB values (< 10ms)
                    if data.value > 0.01:
                        logger.info(f"📊 [{self.service_name}] TTFB: {data.value:.3f}s")
                        self.ttfb_values.append(data.value)
                    else:
                        logger.debug(f"📊 Ignoring spurious TTFB: {data.value:.3f}s")
        elif isinstance(frame, TTSStartedFrame):
            self.start_time = time.time()
            logger.info(f"🎤 [{self.service_name}] TTS Started")
        elif isinstance(frame, TTSStoppedFrame):
            logger.info(f"🎤 [{self.service_name}] TTS Stopped")
        elif isinstance(frame, TTSAudioRawFrame):
            if self.first_audio_time is None:
                self.first_audio_time = time.time()
                elapsed = self.first_audio_time - self.start_time if self.start_time else 0
                logger.info(
                    f"🔊 [{self.service_name}] First audio frame (elapsed: {elapsed:.3f}s, size: {len(frame.audio)} bytes)"
                )
                self.first_chunk_bytes = frame.audio
                self.sample_rate = frame.sample_rate

            # Accumulate all audio chunks
            self.all_audio_chunks.append(frame.audio)
            self.audio_frame_count += 1
            self.total_audio_bytes += len(frame.audio)
            self.last_audio_time = time.time()

        await self.push_frame(frame, direction)

    def _save_wav(self, audio_data: bytes, filename: str):
        """Save audio data to a WAV file."""
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Write as WAV file (16-bit PCM)
        sample_rate = self.sample_rate or 24000
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_data)

        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))  # File size - 8
            f.write(b"WAVE")

            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # Chunk size
            f.write(struct.pack("<H", 1))  # Audio format (PCM)
            f.write(struct.pack("<H", num_channels))
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", byte_rate))
            f.write(struct.pack("<H", block_align))
            f.write(struct.pack("<H", bits_per_sample))

            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(audio_data)

        return filepath

    def save_audio_files(self):
        """Save first chunk and full audio to WAV files."""
        if not self.save_audio:
            return

        safe_name = self.service_name.lower().replace(" ", "_")

        # Save first chunk
        if self.first_chunk_bytes:
            filename = f"{safe_name}_http_first_chunk.wav"
            filepath = self._save_wav(self.first_chunk_bytes, filename)
            logger.info(
                f"💾 [{self.service_name}] Saved first chunk to {filepath} ({len(self.first_chunk_bytes)} bytes)"
            )

        # Save full audio
        if self.all_audio_chunks:
            full_audio = b"".join(self.all_audio_chunks)
            filename = f"{safe_name}_http_full.wav"
            filepath = self._save_wav(full_audio, filename)
            duration_ms = (len(full_audio) / 2 / (self.sample_rate or 24000)) * 1000
            logger.info(
                f"💾 [{self.service_name}] Saved full audio to {filepath} ({len(full_audio)} bytes, {duration_ms:.0f}ms)"
            )

    async def _check_audio_done(self):
        """Check if audio has stopped arriving and signal completion."""
        # Wait for first audio to arrive
        while self.first_audio_time is None:
            await asyncio.sleep(0.1)

        # Now wait for audio to stop (no new audio for 2 seconds)
        while True:
            await asyncio.sleep(0.5)
            if self.last_audio_time and (time.time() - self.last_audio_time) > 2.0:
                logger.info(f"🏁 [{self.service_name}] Audio stream complete")
                # Save audio files when done
                self.save_audio_files()
                self._done_event.set()
                break

    async def wait_for_completion(self):
        """Wait for audio to finish arriving."""
        await self._done_event.wait()

    def get_results(self) -> Dict:
        """Return benchmark results as a dictionary."""
        first_chunk_size = len(self.first_chunk_bytes) if self.first_chunk_bytes else 0

        if self.ttfb_values:
            return {
                "service": self.service_name,
                "ttfb_count": len(self.ttfb_values),
                "ttfb_avg": sum(self.ttfb_values) / len(self.ttfb_values),
                "ttfb_min": min(self.ttfb_values),
                "ttfb_max": max(self.ttfb_values),
                "ttfb_values": self.ttfb_values,
                "audio_frames": self.audio_frame_count,
                "audio_bytes": self.total_audio_bytes,
                "first_chunk_size": first_chunk_size,
            }
        return {
            "service": self.service_name,
            "ttfb_count": 0,
            "ttfb_avg": None,
            "ttfb_min": None,
            "ttfb_max": None,
            "ttfb_values": [],
            "audio_frames": self.audio_frame_count,
            "audio_bytes": self.total_audio_bytes,
            "first_chunk_size": first_chunk_size,
        }


def create_inworld_http_tts(api_key: str, session: aiohttp.ClientSession):
    """Create an Inworld HTTP TTS service without timestamps for fair comparison.
    
    Skips the timestampType parameter in API requests to avoid server-side
    timestamp calculation overhead.
    """
    import base64
    import json
    from typing import AsyncGenerator

    from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame
    from pipecat.services.inworld.tts import InworldHttpTTSService
    from pipecat.utils.tracing.service_decorators import traced_tts

    class InworldHttpTTSServiceNoTimestamps(InworldHttpTTSService):
        """Inworld HTTP TTS without timestamp requests for fair benchmarking."""

        @traced_tts
        async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
            """Generate TTS audio without requesting timestamps."""
            logger.debug(f"{self}: Generating TTS [{text}] (no timestamps)")

            payload = {
                "text": text,
                "voiceId": self._settings["voiceId"],
                "modelId": self._settings["modelId"],
                "audioConfig": self._settings["audioConfig"],
            }

            if "temperature" in self._settings:
                payload["temperature"] = self._settings["temperature"]

            # NOTE: Intentionally NOT including timestampType for fair benchmarking

            headers = {
                "Authorization": f"Basic {self._api_key}",
                "Content-Type": "application/json",
            }

            try:
                await self.start_ttfb_metrics()

                if not self._started:
                    yield TTSStartedFrame()
                    self._started = True

                async with self._session.post(
                    self._base_url, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Inworld API error: {error_text}")
                        yield ErrorFrame(error=f"Inworld API error: {error_text}")
                        return

                    if self._streaming:
                        # Process streaming response
                        buffer = ""
                        async for chunk in response.content.iter_chunked(1024):
                            buffer += chunk.decode("utf-8")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line_str = line.strip()
                                if not line_str:
                                    continue
                                try:
                                    chunk_data = json.loads(line_str)
                                    if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                                        await self.stop_ttfb_metrics()
                                        audio = base64.b64decode(chunk_data["result"]["audioContent"])
                                        # Strip WAV header if present
                                        if len(audio) > 44 and audio.startswith(b"RIFF"):
                                            audio = audio[44:]
                                        if audio:
                                            yield TTSAudioRawFrame(
                                                audio=audio,
                                                sample_rate=self.sample_rate,
                                                num_channels=1,
                                            )
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Non-streaming response
                        response_data = await response.json()
                        if "audioContent" not in response_data:
                            yield ErrorFrame(error="No audioContent in response")
                            return
                        audio_data = base64.b64decode(response_data["audioContent"])
                        if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
                            audio_data = audio_data[44:]
                        chunk_size = self.chunk_size
                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i : i + chunk_size]
                            if chunk:
                                await self.stop_ttfb_metrics()
                                yield TTSAudioRawFrame(
                                    audio=chunk,
                                    sample_rate=self.sample_rate,
                                    num_channels=1,
                                )

                await self.start_tts_usage_metrics(text)

            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            finally:
                await self.stop_all_metrics()

    return InworldHttpTTSServiceNoTimestamps(
        api_key=api_key,
        aiohttp_session=session,
        voice_id="Ashley",
        model="inworld-tts-1.5-max",
        streaming=True,
        aggregate_sentences=True,
    )


def create_elevenlabs_http_tts(api_key: str, session: aiohttp.ClientSession):
    """Create an ElevenLabs HTTP TTS service without timestamps for fair comparison.
    
    Uses the regular /stream endpoint instead of /stream/with-timestamps
    to avoid the overhead of timestamp calculation on the server.
    """
    import base64
    import json
    from typing import AsyncGenerator, Dict, Union

    from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame
    from pipecat.services.elevenlabs.tts import (
        ELEVENLABS_MULTILINGUAL_MODELS,
        ElevenLabsHttpTTSService,
    )
    from pipecat.utils.tracing.service_decorators import traced_tts

    class ElevenLabsHttpTTSServiceNoTimestamps(ElevenLabsHttpTTSService):
        """ElevenLabs HTTP TTS using /stream endpoint (no timestamps) for fair benchmarking."""

        @traced_tts
        async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
            """Generate speech using the regular stream endpoint (no timestamps)."""
            logger.debug(f"{self}: Generating TTS [{text}] (no timestamps)")

            # Use the regular stream endpoint WITHOUT timestamps
            url = f"{self._base_url}/v1/text-to-speech/{self._voice_id}/stream"

            payload: Dict[str, Union[str, Dict[str, Union[float, bool]]]] = {
                "text": text,
                "model_id": self._model_name,
            }

            if self._previous_text:
                payload["previous_text"] = self._previous_text

            if self._voice_settings:
                payload["voice_settings"] = self._voice_settings

            if self._pronunciation_dictionary_locators:
                payload["pronunciation_dictionary_locators"] = [
                    locator.model_dump() for locator in self._pronunciation_dictionary_locators
                ]

            if self._settings["apply_text_normalization"] is not None:
                payload["apply_text_normalization"] = self._settings["apply_text_normalization"]

            language = self._settings["language"]
            if self._model_name in ELEVENLABS_MULTILINGUAL_MODELS and language:
                payload["language_code"] = language

            headers = {
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            }

            params = {
                "output_format": self._output_format,
            }
            if self._settings["optimize_streaming_latency"] is not None:
                params["optimize_streaming_latency"] = self._settings["optimize_streaming_latency"]

            try:
                await self.start_ttfb_metrics()

                async with self._session.post(
                    url, json=payload, headers=headers, params=params
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield ErrorFrame(error=f"ElevenLabs API error: {error_text}")
                        return

                    await self.start_tts_usage_metrics(text)

                    if not self._started:
                        yield TTSStartedFrame()
                        self._started = True

                    # Stream raw audio chunks (no JSON parsing needed)
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(chunk, self.sample_rate, 1)

                    # Update previous text for context
                    if self._previous_text:
                        self._previous_text += " " + text
                    else:
                        self._previous_text = text

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
            finally:
                await self.stop_ttfb_metrics()

    return ElevenLabsHttpTTSServiceNoTimestamps(
        api_key=api_key,
        aiohttp_session=session,
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        model="eleven_turbo_v2_5",
        aggregate_sentences=True,
    )


def create_cartesia_http_tts(api_key: str, session: aiohttp.ClientSession):
    """Create a Cartesia HTTP TTS service."""
    from pipecat.services.cartesia.tts import CartesiaHttpTTSService

    return CartesiaHttpTTSService(
        api_key=api_key,
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady voice
        model="sonic-3",
    )


async def _run_benchmark(llm_simulator, tts, collector):
    """Run the actual benchmark pipeline."""
    pipeline = Pipeline([llm_simulator, tts, collector])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)

    logger.info(f"Starting benchmark for {collector.service_name}...")

    start_time = time.time()

    # Run the pipeline in a task so we can cancel it when audio completes
    run_task = asyncio.create_task(runner.run(task))

    # Wait for audio to complete
    await collector.wait_for_completion()

    # Give a moment for final frames to process
    await asyncio.sleep(0.5)

    # Cancel the pipeline
    await task.cancel()

    # Wait for the runner to finish
    try:
        await asyncio.wait_for(run_task, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Pipeline took too long to cancel, forcing exit")
    except asyncio.CancelledError:
        pass

    total_time = time.time() - start_time
    logger.info(f"Benchmark for {collector.service_name} completed in {total_time:.2f}s")

    return collector.get_results()


async def benchmark_service(
    service_name: str,
    create_tts_fn,
    text: str,
    token_delay_ms: float,
    save_audio: bool = True,
    output_dir: str = "benchmark_audio",
    **create_kwargs,
) -> Dict:
    """
    Run the TTFB benchmark for a specific HTTP-based TTS service.

    Args:
        service_name: Name of the service for logging
        create_tts_fn: Function to create the TTS service
        text: The text to synthesize
        token_delay_ms: Delay between tokens in milliseconds
        save_audio: Whether to save audio files (first chunk + full audio)
        output_dir: Directory to save audio files
        **create_kwargs: Additional keyword arguments for the TTS creation function

    Returns:
        Dictionary with benchmark results
    """
    llm_simulator = SimulatedLLMProcessor(text=text, token_delay_ms=token_delay_ms)
    collector = TTFBCollector(
        service_name=service_name,
        save_audio=save_audio,
        output_dir=output_dir,
    )

    logger.info(f"Using HTTP-based {service_name}")

    async with aiohttp.ClientSession() as session:
        tts = create_tts_fn(session=session, **create_kwargs)
        return await _run_benchmark(llm_simulator, tts, collector)


def print_comparison_table(results: List[Dict]):
    """Print a comparison table of benchmark results."""
    print("\n" + "=" * 80)
    print("HTTP TTS BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(f"{'Service':<20} {'Avg TTFB':<12} {'Min TTFB':<12} {'Max TTFB':<12} {'Samples':<10}")
    print("-" * 80)

    # Sort by average TTFB (fastest first)
    sorted_results = sorted(results, key=lambda x: x.get("ttfb_avg") or float("inf"))

    for r in sorted_results:
        if r["ttfb_avg"] is not None:
            print(
                f"{r['service']:<20} {r['ttfb_avg']:.3f}s{'':<6} {r['ttfb_min']:.3f}s{'':<6} "
                f"{r['ttfb_max']:.3f}s{'':<6} {r['ttfb_count']:<10}"
            )
        else:
            print(f"{r['service']:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {0:<10}")

    print("=" * 80)

    # Winner announcement
    if sorted_results and sorted_results[0]["ttfb_avg"] is not None:
        winner = sorted_results[0]
        print(f"\n🏆 Fastest average TTFB: {winner['service']} ({winner['ttfb_avg']:.3f}s)")

    # Individual sentence breakdown
    print("\n" + "-" * 80)
    print("Per-Sentence TTFB Breakdown:")
    print("-" * 80)

    max_sentences = max(len(r["ttfb_values"]) for r in results if r["ttfb_values"])

    for i in range(max_sentences):
        print(f"\nSentence {i + 1}:")
        sentence_results = []
        for r in results:
            if i < len(r["ttfb_values"]):
                sentence_results.append((r["service"], r["ttfb_values"][i]))

        # Sort by TTFB for this sentence
        sentence_results.sort(key=lambda x: x[1])
        for service, ttfb in sentence_results:
            print(f"  {service:<20} {ttfb:.3f}s")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark TTFB for HTTP-based TTS providers"
    )
    parser.add_argument(
        "--token-delay",
        type=float,
        default=50,
        help="Delay between tokens in milliseconds (default: 50)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text to synthesize",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Number of benchmark iterations to run (default: 1)",
    )
    parser.add_argument(
        "--services",
        type=str,
        default="all",
        help="Comma-separated list of services to benchmark: inworld,elevenlabs,cartesia or 'all' (default: all)",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Disable saving audio files",
    )
    args = parser.parse_args()

    # Default text with multiple sentences
    if args.text is None:
        text = (
            "Hello! Welcome to the TTS benchmark. "
            "This is a test of the text-to-speech system. "
            "Each sentence should trigger a separate TTS request. "
            "Let's see how fast the first audio byte arrives!"
        )
    else:
        text = args.text

    # Parse services to benchmark
    if args.services.lower() == "all":
        services_to_run = ["inworld", "elevenlabs", "cartesia"]
    else:
        services_to_run = [s.strip().lower() for s in args.services.split(",")]

    # Service configurations
    service_configs = {
        "inworld": {
            "name": "Inworld HTTP",
            "create_fn": create_inworld_http_tts,
            "api_key_env": "INWORLD_API_KEY",
            "extra_env": {},
        },
        "elevenlabs": {
            "name": "ElevenLabs HTTP",
            "create_fn": create_elevenlabs_http_tts,
            "api_key_env": "XI_API_KEY",
            "extra_env": {},
        },
        "cartesia": {
            "name": "Cartesia HTTP",
            "create_fn": create_cartesia_http_tts,
            "api_key_env": "CARTESIA_API_KEY",
            "extra_env": {},
        },
    }

    # Check API keys and filter services
    available_services = []
    for service_id in services_to_run:
        if service_id not in service_configs:
            logger.warning(f"Unknown service: {service_id}")
            continue

        config = service_configs[service_id]
        api_key = os.getenv(config["api_key_env"])

        if not api_key:
            logger.warning(f"{config['name']}: {config['api_key_env']} not set, skipping")
            continue

        # Check extra environment variables
        extra_kwargs = {}
        missing_env = False
        for kwarg_name, env_var in config["extra_env"].items():
            value = os.getenv(env_var)
            if not value:
                logger.warning(f"{config['name']}: {env_var} not set, skipping")
                missing_env = True
                break
            extra_kwargs[kwarg_name] = value

        if missing_env:
            continue

        available_services.append((service_id, config, api_key, extra_kwargs))

    if not available_services:
        print("No services available to benchmark. Please set the required API keys:")
        print("  - INWORLD_API_KEY for Inworld HTTP")
        print("  - XI_API_KEY for ElevenLabs HTTP")
        print("  - CARTESIA_API_KEY for Cartesia HTTP")
        return

    print(
        f"\n🚀 Benchmarking {len(available_services)} HTTP TTS service(s): "
        f"{', '.join(c[1]['name'] for c in available_services)}"
    )
    print(f"📝 Text: {text[:50]}..." if len(text) > 50 else f"📝 Text: {text}")
    print(f"⏱️  Token delay: {args.token_delay}ms")
    print(f"🔄 Iterations: {args.iterations}")
    print()

    all_results = {service_id: [] for service_id, _, _, _ in available_services}

    for iteration in range(args.iterations):
        if args.iterations > 1:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} of {args.iterations}")
            print(f"{'='*60}")

        for service_id, config, api_key, extra_kwargs in available_services:
            try:
                result = await benchmark_service(
                    service_name=config["name"],
                    create_tts_fn=config["create_fn"],
                    text=text,
                    token_delay_ms=args.token_delay,
                    save_audio=not args.no_save_audio,
                    api_key=api_key,
                    **extra_kwargs,
                )
                all_results[service_id].append(result)
            except Exception as e:
                logger.error(f"Error benchmarking {config['name']}: {e}")
                import traceback

                traceback.print_exc()
                all_results[service_id].append(
                    {
                        "service": config["name"],
                        "ttfb_count": 0,
                        "ttfb_avg": None,
                        "ttfb_min": None,
                        "ttfb_max": None,
                        "ttfb_values": [],
                        "audio_frames": 0,
                        "audio_bytes": 0,
                        "error": str(e),
                    }
                )

            # Small delay between services
            await asyncio.sleep(1.0)

        # Small delay between iterations
        if iteration < args.iterations - 1:
            await asyncio.sleep(2.0)

    # Aggregate results across iterations
    aggregated_results = []
    for service_id, results_list in all_results.items():
        all_ttfb = []
        for r in results_list:
            all_ttfb.extend(r["ttfb_values"])

        if all_ttfb:
            aggregated_results.append(
                {
                    "service": results_list[0]["service"],
                    "ttfb_count": len(all_ttfb),
                    "ttfb_avg": sum(all_ttfb) / len(all_ttfb),
                    "ttfb_min": min(all_ttfb),
                    "ttfb_max": max(all_ttfb),
                    "ttfb_values": all_ttfb,
                }
            )
        else:
            aggregated_results.append(
                {
                    "service": service_configs[service_id]["name"],
                    "ttfb_count": 0,
                    "ttfb_avg": None,
                    "ttfb_min": None,
                    "ttfb_max": None,
                    "ttfb_values": [],
                }
            )

    # Print comparison
    print_comparison_table(aggregated_results)

    # Print aggregate stats if multiple iterations
    if args.iterations > 1:
        print("\n" + "=" * 80)
        print(f"AGGREGATE STATISTICS ({args.iterations} iterations)")
        print("=" * 80)
        for r in sorted(aggregated_results, key=lambda x: x.get("ttfb_avg") or float("inf")):
            if r["ttfb_avg"] is not None:
                # Calculate std dev
                mean = r["ttfb_avg"]
                variance = sum((x - mean) ** 2 for x in r["ttfb_values"]) / len(r["ttfb_values"])
                std_dev = variance**0.5
                print(f"\n{r['service']}:")
                print(f"  Total samples: {r['ttfb_count']}")
                print(f"  Average TTFB:  {r['ttfb_avg']:.3f}s")
                print(f"  Min TTFB:      {r['ttfb_min']:.3f}s")
                print(f"  Max TTFB:      {r['ttfb_max']:.3f}s")
                print(f"  Std Dev:       {std_dev:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
