import psutil
import time, faiss
from faiss import IndexFlatL2
import numpy as np
from ..query.indexdb import IndexDB
from .load_conf import load_conf
import pickle
from transformers import AutoConfig, AutoModel
from .check_device import check_device


# Example FAISS indexing operation
def faiss_example():
    config = load_conf()

    with open("data/embeddings.pkl", "rb") as f:
        data1 = pickle.load(f)

    with open("data/chunks.pkl", "rb") as f:
        data2 = pickle.load(f)

    check_device()

    embeddings_config = AutoConfig.from_pretrained(config["Paths"]["embeddings_path"])
    hidden_size = embeddings_config.hidden_size
    print(hidden_size)

    check_device()
    np.random.seed(0)
    query = np.random.random((10, hidden_size)).astype('float32')


    index = faiss.IndexFlatL2(hidden_size)

    check_device()
    faiss_memory_start = psutil.Process().memory_info().rss

    index.add(data1["embeddings"])

    check_device()
    faiss_memory_end = psutil.Process().memory_info().rss

    faiss_overhead = faiss_memory_end - faiss_memory_start
    print(f"FAISS memory overhead: {faiss_overhead / (1024 ** 2):.2f} MB")


    D, I = index.search(query, config["General"]["k-nearest"])





# Monitor CPU usage
def monitor_cpu(function, interval=0.1):
    process = psutil.Process()  # Get the current process
    cpu_usage = []

    # Monitor CPU while the function runs
    start_time = time.time()
    function()  # Run your function
    elapsed_time = time.time() - start_time

    # Poll CPU usage at intervals
    for _ in range(int(elapsed_time / interval)):
        cpu_usage.append(process.cpu_percent(interval=interval))

    print(f"Average CPU Usage: {sum(cpu_usage) / len(cpu_usage):.2f}%")
    print(f"Peak CPU Usage: {max(cpu_usage):.2f}%")

# Run the FAISS example with monitoring
if __name__ == "__main__":

    monitor_cpu(faiss_example)
