# web_RAG

This repository demonstrates a proof of concept for using a Retrieval-Augmented Generation (RAG) system as a helper bot for websites. The system is designed to assist users in retrieving information and navigating websites effectively.

The project explores the feasibility of building a website assistant bot tailored for the University of Luxembourg. It was developed under the supervision of Prof. Martin Theobald.

---

## Pre-Requisites

### Hardware Requirements
This project was developed on a MacBook Pro M1 running macOS Sequoia. Due to memory limitations, the retrieval, chunking, and embedding processes were outsourced to a virtual machine (VM) with higher computational resources. 

**Minimum Requirements:**
- Modern device capable of running Python 3.8 or newer
- At least **16GB RAM** (adjustable based on the website size)

While this project is optimized for the mentioned hardware and OS, it should run on other modern setups with minimal adjustments.

### Software Environment
1. Install dependencies listed in `requirements.txt`.
2. Use the default settings in the provided configuration file (`config`).

---

## Installation

### 1. Clone the Repository
Clone the repository to your local machine:

```bash
# Using HTTPS
git clone https://github.com/TarikTornes/web_RAG.git

# Using SSH
git clone git@github.com:TarikTornes/web_RAG.git
```

### 2. Install dependencies
Set up a Python environment and install dependencies:
```bash
# Example using pip
pip install -r requirements.txt
```

### 3. Install a Language Model
Download and install a quantized language model for optimal performance. We recommend the **Meta Llama 3.1 8B Instruct** model.
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --local-dir ./data
```

Ensure the .gguf file is placed in the /data directory.

### 4. Configure settings
Update the configuration file (`config.yaml`) to match your environment and data setup.


## Running the script
1. Navigate to the root directory of the repository.
2. Execute the following script to start the RAG system:

```bash
./scripts/run_all.sh
```

### First-Time Setup
- During the first run, you will be prompted to execute the chunking and embedding scripts. These steps must be completed at least once to process and save the website data.
- Note: Depending on the website's size and your hardware, these steps may take significant time and resources. For large-scale websites, it is recommended to:
    - Run the chunking and embedding scripts on a powerful VM with multiple cores.
    - Allow the process to run overnight if necessary.
    - Once complete, copy the generated pickled files to the /data directory on your local machine.
