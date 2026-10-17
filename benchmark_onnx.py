"""
Benchmark ONNX Runtime latency for Edge / CPU inference
========================================================
Usage:
  python benchmark_onnx.py
  python benchmark_onnx.py --model "runs/detect/vn_direct/weights/best.onnx" --runs 200
"""
import os
import time
import argparse
import numpy as np
import onnxruntime as ort


def find_default_onnx():
    candidates = [
        os.path.join("runs", "detect", "vn_direct", "weights", "best.onnx"),
        os.path.join("weights", "best_yolo_vn.onnx"),
        os.path.join("weights", "best_colab_old.onnx"),
        "yolo11n.onnx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX runtime model")
    parser.add_argument(
        "--model",
        type=str,
        default=find_default_onnx(),
        help="Path to .onnx file",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Input dimension")
    args = parser.parse_args()

    if not args.model or not os.path.exists(args.model):
        print("⚠️ No ONNX model found. Please run export_onnx.py first!")
        print("   Example: python export_onnx.py")
        return

    print(f"⏱️ Benchmarking ONNX: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)

    # Warm-up
    print("   Warming up...")
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Benchmark
    print(f"   Running {args.runs} benchmark iterations...")
    t0 = time.time()
    for _ in range(args.runs):
        session.run(None, {input_name: dummy})
    t1 = time.time()

    avg_latency = (t1 - t0) / args.runs * 1000
    fps = 1000 / avg_latency
    print(f"\n📊 Results:")
    print(f"   Average CPU Latency: {avg_latency:.2f} ms")
    print(f"   Estimated Throughput: {fps:.1f} FPS")


if __name__ == "__main__":
    main()
