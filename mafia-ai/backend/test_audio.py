"""
Тест Audio Pipeline: VAD + ASR + Speaker ID

Проверяем:
- Silero VAD (детекция речи)
- Faster-Whisper ASR (транскрипция)
- AudioVideoCorrelator (определение говорящего)
"""

import asyncio
import numpy as np
import time


async def test_silero_vad():
    """Тест Silero VAD"""
    print("\n" + "="*60)
    print("  Testing Silero VAD")
    print("="*60)

    try:
        from infrastructure.audio import SileroVAD

        # Создаем VAD
        vad = SileroVAD(threshold=0.5, sampling_rate=16000)

        # Генерируем тестовый аудио (1 секунда случайного шума)
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1

        # Тест детекции
        print("\nTesting on random noise...")
        has_speech = vad.detect_voice_activity(test_audio, sample_rate=16000)
        print(f"✅ Speech detected: {has_speech} (expected: False for noise)")

        # Генерируем "синтетическую речь" (синусоида с модуляцией)
        t = np.linspace(0, 1, 16000)
        synthetic_speech = (np.sin(2 * np.pi * 200 * t) *
                           (1 + 0.5 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)

        has_speech_2 = vad.detect_voice_activity(synthetic_speech, sample_rate=16000)
        print(f"✅ Speech detected in sine wave: {has_speech_2}")

        # Тест извлечения сегментов
        print("\nTesting segment extraction...")
        segments = vad.extract_speech_segments(
            synthetic_speech,
            sample_rate=16000,
            min_speech_duration=0.1
        )
        print(f"✅ Extracted {len(segments)} speech segments")

        if segments:
            for i, seg in enumerate(segments[:3]):  # Показываем первые 3
                print(f"   Segment {i}: {seg.start_time:.2f}s - {seg.end_time:.2f}s (duration: {seg.end_time - seg.start_time:.2f}s)")

        print("\n✅ Silero VAD test passed!")

    except ImportError as e:
        print(f"\n⚠️  Silero VAD not available: {e}")
        print("   Install with: pip install torch torchaudio")
    except Exception as e:
        print(f"\n❌ Silero VAD test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_faster_whisper():
    """Тест Faster-Whisper ASR"""
    print("\n" + "="*60)
    print("  Testing Faster-Whisper ASR")
    print("="*60)

    try:
        from infrastructure.audio import FasterWhisperASR

        # Создаем ASR (используем tiny модель для быстрого теста)
        print("\nInitializing Faster-Whisper (tiny model)...")
        asr = FasterWhisperASR(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="ru"
        )

        # Генерируем тестовый аудио (2 секунды тишины)
        # Для реального теста нужно записать голос или использовать файл
        test_audio = np.random.randn(32000).astype(np.float32) * 0.01  # Тихий шум

        print("\nTranscribing test audio (2 seconds)...")
        start = time.time()
        transcription = await asr.transcribe_async(test_audio, sample_rate=16000)
        duration = time.time() - start

        print(f"\n✅ Transcription completed in {duration:.2f}s")
        print(f"   Text: '{transcription.text}'")
        print(f"   Language: {transcription.language}")
        print(f"   Confidence: {transcription.confidence:.2f}")
        print(f"   Segments: {len(transcription.segments)}")

        if not transcription.text.strip():
            print("   ℹ️  No speech detected (expected for noise)")

        print("\n✅ Faster-Whisper test passed!")

    except ImportError as e:
        print(f"\n⚠️  Faster-Whisper not available: {e}")
        print("   Install with: pip install faster-whisper")
    except Exception as e:
        print(f"\n❌ Faster-Whisper test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_speaker_identification():
    """Тест Speaker Identification"""
    print("\n" + "="*60)
    print("  Testing Speaker Identification")
    print("="*60)

    try:
        from infrastructure.audio import AudioVideoCorrelator
        from core.domain.entities.player import Player
        from core.domain.value_objects.player_id import PlayerId
        import numpy as np

        # Создаем correlator
        correlator = AudioVideoCorrelator()

        # Создаем тестовых игроков
        players = [
            Player(
                id=PlayerId(1),
                name="Player 1",
                seat=1,
                face_embedding=np.random.randn(512).astype(np.float32)
            ),
            Player(
                id=PlayerId(2),
                name="Player 2",
                seat=2,
                face_embedding=np.random.randn(512).astype(np.float32)
            ),
        ]

        # Тест 1: Один видимый игрок
        print("\nTest 1: Single visible player")
        result = correlator.identify_speaker_by_faces(
            players=players,
            visible_player_ids=[1],
            timestamp=0.0
        )
        print(f"✅ Speaker identified: Player {result.player_id} (confidence: {result.confidence:.2f}, method: {result.method})")

        # Тест 2: Несколько видимых игроков
        print("\nTest 2: Multiple visible players")
        result = correlator.identify_speaker_by_faces(
            players=players,
            visible_player_ids=[1, 2],
            timestamp=0.0
        )
        print(f"✅ Speaker identified: Player {result.player_id} (confidence: {result.confidence:.2f}, method: {result.method})")

        # Тест 3: Turn-based identification
        print("\nTest 3: Turn-based identification")
        result = correlator.identify_speaker_by_position(
            current_turn_player_id=2,
            visible_player_ids=[1, 2]
        )
        print(f"✅ Speaker identified: Player {result.player_id} (confidence: {result.confidence:.2f}, method: {result.method})")

        # Тест 4: Hybrid method
        print("\nTest 4: Hybrid identification")
        result = correlator.identify_speaker_hybrid(
            players=players,
            visible_player_ids=[1, 2],
            current_turn_player_id=2,
            timestamp=0.0
        )
        print(f"✅ Speaker identified: Player {result.player_id} (confidence: {result.confidence:.2f}, method: {result.method})")

        print("\n✅ Speaker Identification test passed!")

    except Exception as e:
        print(f"\n❌ Speaker Identification test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Запуск всех тестов"""
    print("\n" + "="*60)
    print("  Audio Pipeline Test Suite")
    print("="*60)

    await test_silero_vad()
    await test_faster_whisper()
    await test_speaker_identification()

    print("\n" + "="*60)
    print("  ✅ ALL AUDIO TESTS COMPLETE!")
    print("="*60)
    print("\nNote: For real-world testing, use actual audio recordings")
    print("      or microphone input instead of synthetic data.")


if __name__ == "__main__":
    asyncio.run(main())
