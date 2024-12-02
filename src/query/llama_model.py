from accelerate import Accelerator
from llama_cpp import Llama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Llama_model:

    def __init__(self, path):
        model = Llama(
            model_path=path,
            n_gpu_layers=-1,
            max_tokens=8192,
            n_ctx=8192,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

        print("TESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSST")

        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(model)

    def alpaca_prompt(self, instruction, input=None, output=None):
        if input and output:    
            return (
            "Below is an instruction that describes a task, \
            paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        )
        elif input:
            return (
                "Below is an instruction that describes a task, \
                paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )
        elif output:
            return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        )
        else:
            return (
            "Below is an instruction that describes a task, \
            paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"
        )

    def getAnswer(self, query_results, INSTRUCTION, MAX_TOKENS=1024):
        alpaca = self.alpaca_prompt(INSTRUCTION, input='\n'.join(result[1] for result in query_results))
        answer = self.model(alpaca, max_tokens=MAX_TOKENS)
        print("\n------------------------------------------\n")
        return answer['choices'][0]['text']


