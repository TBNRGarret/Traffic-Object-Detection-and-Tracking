import time
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession(
    r"D:\FPT\runs\detect\train5\weights\best.onnx",
    providers=["CPUExecutionProvider"]
)

# Dummy input (batch=1)
input_name = session.get_inputs()[0].name
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Warm-up
for _ in range(10):
    session.run(None, {input_name: dummy})

# Benchmark
runs = 100
t0 = time.time()
for _ in range(runs):
    session.run(None, {input_name: dummy})
t1 = time.time()

avg_latency = (t1 - t0) / runs * 1000
print(f"Average ONNX CPU latency: {avg_latency:.2f} ms")
