from accelerate import Accelerator
from llama_cpp import Llama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ..utils.instruction_format import instruction_format


class Llama_model:

    def __init__(self, path):
        model = Llama(
            model_path=path,
            n_gpu_layers=-1,
            max_tokens=8192,
            n_ctx=12000,
            f16_kv=True,
            verbose=False
            #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
                    f"When you receive an instruction, use the Input to generate an answer to the original user question.\n"
                    f"The Input consists of the most relevant documents for the given instruction and their source URL after the\n"
                    f"[\"FROM:\"] token.\n\n"
                    f"You are a helpful assistant serving as a website bot.<|eot_id|>\n"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"Please reply to the following instruction in a concise manner and"
                    f"provide exactly one URL where the most important information can be found.\n\n"
                    f" If the input does not contain the exactly correct information, reply in the format:\n"
                    f"Sorry I could not reply to your question maybe you can find some information on [\"URL\"]\n\n"
                    f"### Input:\n{input}\n\n"
                    f"### Instruction:\n{instruction}<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>"
                )

            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|>"
                )
            
        return formatted_prompt


    def get_formatted_prompt3(self, instruction, input=None, output=None):

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
                    f"<|start_header_id|>system<|end_header_id|>\n\n"
                    f"You are a knowledgeable assistant helping users find accurate information from a website. Your role is to:\n"
                    f"1. Provide clear, direct answers based on the provided documents\n"
                    f"2. Ensure responses are factual and grounded in the source material\n"
                    f"3. Maintain a professional and helpful tone\n\n"
                    f"Guidelines:\n"
                    f"- Focus on the most relevant information from the Input documents\n"
                    f"- Include always exactly one URL for the primary source\n"
                    f"- Do not reference the Input documents directly in your response\n"
                    f"- Keep responses concise and to the point\n"
                    f"- If information is insufficient, use the error response template<|eot_id|>\n\n"
                    
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"Answer the following instruction using only information from the provided documents.\n\n"
                    f"Response format for complete and unibiased information:\n"
                    f"[Concise answer addressing the instruction]\n"
                    f"Learn more: [URL]\n\n"
                    f"Response format for unclear or insufficient information:\n"
                    f"Sorry, I don't have enough information to fully answer your question. You may find relevant details here: [URL]\n\n"
                    
                    f"### Input:\n{input}\n\n"
                    f"### Instruction:\n{instruction}<|eot_id|>\n"
                    
                    f"<|start_header_id|>assistant<|end_header_id|>"
                )

            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|>"
                )
            
        return formatted_prompt






    def getAnswer(self, query_results, INSTRUCTION, MAX_TOKENS=1024):
        input = instruction_format(query_results)

        alpaca = self.get_formatted_prompt3(INSTRUCTION, input=input)

        answer = self.model(alpaca, max_tokens=MAX_TOKENS)
        print("\n------------------------------------------\n")
        return answer['choices'][0]['text']


