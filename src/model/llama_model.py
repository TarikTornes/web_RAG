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
            n_ctx=12000,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )


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


    # Prompt format from official Meta Llama-3 documentation: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    def get_formatted_prompt2(self, instruction, input=None, output=None):

        if output:
            if not input:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    f"{output}"
                    f"<|eot_id|><|end_of_text|>"
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    f"{output}"
                    f"<|eot_id|><|end_of_text|>"
            )
        else:
            if input:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                    f"<|eot_id|>"
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|>"
                )
            
        return formatted_prompt



    def get_formatted_prompt(self, instruction, input=None, output=None):

        if output:
            if not input:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    f"{output}"
                    f"<|eot_id|><|end_of_text|>"
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    f"{output}"
                    f"<|eot_id|><|end_of_text|>"
            )
        else:
            if input:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"You are an assistant serving as a website bot. You will be given an Instruction and an Input text from the website from which you need to reply to the instruction in a concise manner. If you cannot deduce the answer from the given text reply with: \"I am not able to retrieve the necessary information.\". \n<|eot_id|><|start_header_id|>user<|end_header_id|>Instruction:\n{instruction}\n\n Input:\n{input}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|>"
                )
            
        return formatted_prompt





    def getAnswer(self, query_results, INSTRUCTION, MAX_TOKENS=1024):
        alpaca = self.get_formatted_prompt(INSTRUCTION, input='\n'.join(result[1] for result in query_results))
        answer = self.model(alpaca, max_tokens=MAX_TOKENS)
        print("\n------------------------------------------\n")
        return answer['choices'][0]['text']


