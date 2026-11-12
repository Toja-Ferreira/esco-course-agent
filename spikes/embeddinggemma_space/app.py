"""Check: does EmbeddingGemma download and run on CPU in a free Hugging Face Space?

Only the QUERY (one course description) is embedded here at runtime; the 13,939
skill embeddings will be precomputed offline. So this measures:
  1. gated download works with the HF_TOKEN Space secret (needs read access to the model)
  2. model loads within the Space's RAM
  3. latency of embedding one course-length text on CPU
  4. the vector, so check_embedding_parity.py can compare it to your machine's

Endpoints: /embed (vector, CPU), /benchmark (CPU timings), /benchmark_gpu.
ZeroGPU refuses to start without at least one @spaces.GPU function, so the GPU
benchmark is there to satisfy that, and as a comparison point.
"""
import spaces  # ZeroGPU: must be imported before torch

import platform
import statistics
import time

import gradio as gr
import numpy as np
import psutil
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-300m"

# Copied from esco-skill-linker src/retrieval/dense.py (ModelSpec "embeddinggemma"
# + DenseIndex._encode), so queries match the precomputed skill vectors.
# Keep embed() identical in check_embedding_parity.py.
QUERY_PREFIX = "task: search result | query: "


def embed(model, text: str) -> np.ndarray:
    vec = model.encode(
        [QUERY_PREFIX + text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vec, dtype=np.float32)


SAMPLE = (
    "PLACEHOLDER, replace with a real course page. Databases (6 ECTS). "
    "Objectives: students learn to design relational database schemas, apply "
    "normalisation, and write SQL queries including joins, aggregation and "
    "subqueries. Syllabus: the relational model; entity-relationship modelling; "
    "functional dependencies and normal forms; SQL data definition and "
    "manipulation; indexing and query processing; transactions and concurrency "
    "control; building small data pipelines in Python with a database driver. "
    "Assessment: a group project designing and implementing a database for a "
    "real organisation, and a written exam."
)

proc = psutil.Process()
t0 = time.perf_counter()
model = SentenceTransformer(MODEL_ID, device="cpu")  # reads HF_TOKEN from env
LOAD_S = time.perf_counter() - t0
RSS_AFTER_LOAD_MB = proc.memory_info().rss / 2**20


def embed_api(text: str) -> list[float]:
    return embed(model, text).tolist()


def _time_ms(text: str, runs: int = 10) -> list[float]:
    embed(model, text)  # warm-up
    times = []
    for _ in range(int(runs)):
        t = time.perf_counter()
        embed(model, text)
        times.append((time.perf_counter() - t) * 1000)
    return sorted(times)


@spaces.GPU(duration=60)
def benchmark_gpu(text: str) -> dict:
    model.to("cuda")
    try:
        times = _time_ms(text)
        gpu = torch.cuda.get_device_name(0)
    finally:
        model.to("cpu")  # CPU endpoints keep working afterwards
    return {
        "device": gpu,
        "latency_ms_median": round(statistics.median(times), 1),
        "note": "excludes ZeroGPU queue/attach time, which the user would also wait for",
    }


def benchmark(text: str) -> dict:
    times = _time_ms(text)
    return {
        "device": "cpu",
        "model": MODEL_ID,
        "n_tokens": len(model.tokenizer(QUERY_PREFIX + text)["input_ids"]),
        "n_chars": len(text),
        "load_seconds_incl_download": round(LOAD_S, 1),
        "latency_ms_median": round(statistics.median(times), 1),
        "latency_ms_p90": round(times[int(0.9 * (len(times) - 1))], 1),
        "rss_mb_after_load": round(RSS_AFTER_LOAD_MB),
        "rss_mb_now": round(proc.memory_info().rss / 2**20),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch_threads": torch.get_num_threads(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
    }


with gr.Blocks(title="EmbeddingGemma CPU check") as demo:
    gr.Markdown("## EmbeddingGemma on CPU: load, latency, parity")
    text = gr.Textbox(value=SAMPLE, lines=10, label="Course description")
    with gr.Row():
        bench_btn = gr.Button("Benchmark (CPU)")
        gpu_btn = gr.Button("Benchmark (GPU)")
        embed_btn = gr.Button("Embed")
    out = gr.JSON(label="Result")
    bench_btn.click(benchmark, inputs=text, outputs=out, api_name="benchmark")
    gpu_btn.click(benchmark_gpu, inputs=text, outputs=out, api_name="benchmark_gpu")
    embed_btn.click(embed_api, inputs=text, outputs=out, api_name="embed")

if __name__ == "__main__":
    demo.launch()