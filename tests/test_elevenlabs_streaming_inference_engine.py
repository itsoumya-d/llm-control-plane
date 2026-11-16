import pytest
from llm_control_plane.elevenlabs_streaming_inference_engine import (
    ElevenLabsStreamingInferenceEngine,
    InferencePriority,
    KVCachePool,
    QuantizationFormat,
    StreamingInferenceConfig,
)


def test_kv_cache_allocation_and_reuse():
    pool = KVCachePool(max_blocks=10, block_size=8)
    prompt = "Hello, this is a conversational turn with ElevenLabs voice agent."

    blocks_1, hit_1 = pool.allocate("session_01", prompt)
    assert not hit_1
    assert len(blocks_1) > 0
    assert pool.misses == 1

    # Second allocation with same prompt prefix -> hit
    blocks_2, hit_2 = pool.allocate("session_02", prompt)
    assert hit_2
    assert blocks_2 == blocks_1
    assert pool.hits == 1

    pool.release("session_01")
    pool.release("session_02")


def test_quantization_calibration():
    engine_int8 = ElevenLabsStreamingInferenceEngine(
        StreamingInferenceConfig(quantization=QuantizationFormat.INT8)
    )
    res_int8 = engine_int8.calibrate_quantization((4096, 4096))
    assert res_int8["format"] == "INT8"
    assert res_int8["compression_ratio"] == 2.0
    assert res_int8["audio_fidelity_preserved"] is True
    assert res_int8["estimated_snr_db"] > 45.0

    engine_fp8 = ElevenLabsStreamingInferenceEngine(
        StreamingInferenceConfig(quantization=QuantizationFormat.FP8)
    )
    res_fp8 = engine_fp8.calibrate_quantization((4096, 4096))
    assert res_fp8["format"] == "FP8"
    assert res_fp8["compression_ratio"] == 2.0
    assert res_fp8["audio_fidelity_preserved"] is True

    engine_fp16 = ElevenLabsStreamingInferenceEngine(
        StreamingInferenceConfig(quantization=QuantizationFormat.FP16)
    )
    res_fp16 = engine_fp16.calibrate_quantization((4096, 4096))
    assert res_fp16["compression_ratio"] == 1.0


def test_streaming_generation_flow():
    config = StreamingInferenceConfig(
        model_id="eleven_multilingual_v2",
        sample_rate=24000,
        chunk_duration_ms=20.0,
    )
    engine = ElevenLabsStreamingInferenceEngine(config)
    chunks = list(
        engine.generate_streaming(
            prompt="Streaming speech synthesis for real-time customer agent.",
            voice_id="adam_voice_42",
            priority=InferencePriority.INTERACTIVE,
            simulated_token_count=8,
        )
    )

    assert len(chunks) == 8
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.sample_rate == 24000
        assert c.duration_ms == 20.0
        assert len(c.audio_bytes) > 0
        assert c.time_to_chunk_ms > 0
        if idx == 7:
            assert c.is_final is True
        else:
            assert c.is_final is False


def test_profiler_and_sla_metrics():
    config = StreamingInferenceConfig(
        target_ttfb_ms=85.0,
        quantization=QuantizationFormat.INT8,
    )
    engine = ElevenLabsStreamingInferenceEngine(config)
    metrics = engine.profile_request(
        prompt="Testing end-to-end inference SLA metrics.",
        voice_id="bella_voice_03",
        priority=InferencePriority.INTERACTIVE,
        simulated_token_count=10,
    )

    assert metrics.total_chunks == 10
    assert metrics.ttfb_ms > 0
    assert metrics.tokens_per_second > 0
    assert metrics.p50_chunk_ms > 0
    assert metrics.p90_chunk_ms >= metrics.p50_chunk_ms
    assert metrics.p99_chunk_ms >= metrics.p90_chunk_ms
    assert metrics.sla_passed is True


def test_priority_scheduling_impact():
    engine = ElevenLabsStreamingInferenceEngine()
    prompt_interactive = "Priority scheduling latency comparison for interactive voice."
    prompt_batch = "Priority scheduling latency comparison for batch asynchronous."

    interactive_metrics = engine.profile_request(
        prompt=prompt_interactive,
        priority=InferencePriority.INTERACTIVE,
        simulated_token_count=5,
    )
    batch_metrics = engine.profile_request(
        prompt=prompt_batch,
        priority=InferencePriority.BATCH,
        simulated_token_count=5,
    )

    assert interactive_metrics.ttfb_ms < batch_metrics.ttfb_ms
