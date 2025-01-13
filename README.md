# web_RAG

This repository demonstrates a proof of concept for using a Retrieval-Augmented Generation (RAG) system as a helper bot for websites. 
The system is designed to assist users in retrieving information and navigating websites effectively.

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

---

## Setup the Bot
1. Navigate to the root directory of the repository.
2. Run `make` to create executables out of the bash scripts
3. Run the following script to setup the system for your website.
Depending on the size of the website the download might take 
some time. If you have already downloaded it before, you can move it to
the directory `/data/webpages/` and confirm with <n> that you already 
have the webpages downloaded:
    ```shell
    # To download and/or extract content from HTML files
    ./scrits/setup.sh
    ```
 
4. Run the following to create the necessary files for your RAG system:
    ```shell
    # To chunk and embed the website content
    ./scripts/run_all.sh
    ```

--- 


## Running the Bot
Make sure you have successfully completed each of the four **Installation** steps.
Now you can run your personal Website-Chatbot with:
```shell
python3 -m src.main
```

---

## Future Work
- Implement more sophisticated chunking techniques:
    - Graph-Based
    - Structure-Based
    - Hybrid approaches
- Use different reranking techniques:
    - RAGFusion
    - MMR
    - Re2G
- Implement a simple way to connect to APIs like groq


---


## Examples
