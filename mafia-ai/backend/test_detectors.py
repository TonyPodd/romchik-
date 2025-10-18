"""
Тест сравнения детекторов: Legacy vs Hybrid

Сравниваем:
- Legacy: MediaPipe + ArcFace ONNX
- Hybrid: YOLOv8 + InsightFace

Метрики:
- FPS (скорость)
- Количество обнаруженных лиц
- Качество эмбеддингов
"""

import asyncio
import cv2
import time
import numpy as np

from infrastructure.detection.face import LegacyFaceDetector, HybridFaceDetector


async def test_detector_on_camera(detector, detector_name: str, num_frames: int = 100):
    """
    Тест детектора на реальной камере

    Args:
        detector: Детектор для теста
        detector_name: Имя для логов
        num_frames: Количество кадров для теста
    """
    print(f"\n{'='*60}")
    print(f"  Testing {detector_name}")
    print('='*60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return

    # Настройки камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            await detector.detect(frame)

    # Тест
    print(f"Testing on {num_frames} frames...")

    faces_detected = []
    processing_times = []
    frame_count = 0

    start_time = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # Детекция
        faces = await detector.detect(frame)

        frame_end = time.time()

        faces_detected.append(len(faces))
        processing_times.append(frame_end - frame_start)

        # Визуализация (каждый 10й кадр)
        if frame_count % 10 == 0:
            display_frame = frame.copy()
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_text = f"{face.confidence:.2f}"
                cv2.putText(display_frame, conf_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display_frame, f"{detector_name} - Frame {frame_count}/{num_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Faces: {len(faces)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(f'{detector_name} Test', display_frame)
            cv2.waitKey(1)

        frame_count += 1

    total_time = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()

    # Статистика
    avg_faces = np.mean(faces_detected)
    avg_time = np.mean(processing_times) * 1000  # ms
    fps = frame_count / total_time

    print(f"\n📊 Results for {detector_name}:")
    print(f"   Total frames: {frame_count}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average FPS: {fps:.2f}")
    print(f"   Avg processing time: {avg_time:.2f}ms per frame")
    print(f"   Avg faces detected: {avg_faces:.2f}")
    print(f"   Max faces in frame: {max(faces_detected)}")
    print(f"   Min faces in frame: {min(faces_detected)}")

    return {
        "name": detector_name,
        "fps": fps,
        "avg_time_ms": avg_time,
        "avg_faces": avg_faces,
        "max_faces": max(faces_detected),
        "min_faces": min(faces_detected),
    }


async def compare_on_single_frame(frame: np.ndarray):
    """Сравнение на одном кадре"""
    print("\n" + "="*60)
    print("  Single Frame Comparison")
    print("="*60)

    # Legacy
    legacy = LegacyFaceDetector()
    start = time.time()
    legacy_faces = await legacy.detect(frame)
    legacy_time = (time.time() - start) * 1000

    print(f"\n🔵 Legacy Detector:")
    print(f"   Faces detected: {len(legacy_faces)}")
    print(f"   Processing time: {legacy_time:.2f}ms")
    print(f"   Embedding dim: {legacy.get_embedding_dim()}")

    # Hybrid (если доступен)
    try:
        hybrid = HybridFaceDetector(device="cpu")
        start = time.time()
        hybrid_faces = await hybrid.detect(frame)
        hybrid_time = (time.time() - start) * 1000

        print(f"\n🟢 Hybrid Detector:")
        print(f"   Faces detected: {len(hybrid_faces)}")
        print(f"   Processing time: {hybrid_time:.2f}ms")
        print(f"   Embedding dim: {hybrid.get_embedding_dim()}")

        print(f"\n📈 Speedup: {legacy_time / hybrid_time:.2f}x")

    except Exception as e:
        print(f"\n⚠️  Hybrid detector not available: {e}")
        print("   (Install: pip install ultralytics insightface)")


async def main():
    """Главная функция"""
    print("\n" + "="*60)
    print("  Face Detector Comparison Test")
    print("="*60)

    # Проверяем камеру
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available. Exiting.")
        return

    ret, test_frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not capture frame. Exiting.")
        return

    print("✅ Camera available")

    # Сравнение на одном кадре
    await compare_on_single_frame(test_frame)

    # Выбор детектора для полного теста
    print("\n" + "="*60)
    print("Choose detector for full test:")
    print("1. Legacy (MediaPipe + ArcFace ONNX)")
    print("2. Hybrid (YOLOv8 + InsightFace)")
    print("3. Both (compare)")
    print("="*60)

    choice = input("Enter choice (1/2/3) or press Enter to skip: ").strip()

    if choice == "1":
        legacy = LegacyFaceDetector()
        await test_detector_on_camera(legacy, "Legacy", num_frames=100)

    elif choice == "2":
        try:
            hybrid = HybridFaceDetector(device="cpu")
            await test_detector_on_camera(hybrid, "Hybrid", num_frames=100)
        except Exception as e:
            print(f"❌ Hybrid detector error: {e}")

    elif choice == "3":
        legacy = LegacyFaceDetector()
        legacy_results = await test_detector_on_camera(legacy, "Legacy", num_frames=100)

        try:
            hybrid = HybridFaceDetector(device="cpu")
            hybrid_results = await test_detector_on_camera(hybrid, "Hybrid", num_frames=100)

            # Сравнение
            print("\n" + "="*60)
            print("  Final Comparison")
            print("="*60)
            print(f"\n{'Metric':<20} {'Legacy':>15} {'Hybrid':>15} {'Winner':>15}")
            print("-"*70)
            print(f"{'FPS':<20} {legacy_results['fps']:>15.2f} {hybrid_results['fps']:>15.2f} {'🟢 Hybrid' if hybrid_results['fps'] > legacy_results['fps'] else '🔵 Legacy':>15}")
            print(f"{'Avg Time (ms)':<20} {legacy_results['avg_time_ms']:>15.2f} {hybrid_results['avg_time_ms']:>15.2f} {'🟢 Hybrid' if hybrid_results['avg_time_ms'] < legacy_results['avg_time_ms'] else '🔵 Legacy':>15}")
            print(f"{'Avg Faces':<20} {legacy_results['avg_faces']:>15.2f} {hybrid_results['avg_faces']:>15.2f} {'🟢 Hybrid' if hybrid_results['avg_faces'] > legacy_results['avg_faces'] else '🔵 Legacy':>15}")

        except Exception as e:
            print(f"❌ Hybrid test failed: {e}")

    print("\n✅ Tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
