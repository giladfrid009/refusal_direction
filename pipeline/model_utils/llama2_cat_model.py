import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from pipeline.model_utils.model_base import ModelBase

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

LLAMA2_CAT_CHAT_TEMPLATE = "[INST] {instruction} [/INST] "

LLAMA2_CAT_CHAT_TEMPLATE_WITH_SYSTEM = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] "

LLAMA2_CAT_REFUSAL_TOKS = [306, 8221]  # [I, Sorry]


def format_instruction_llama2_chat(instruction: str, output: str = None, system: str = None, include_trailing_whitespace: bool = True):
    if system is not None:
        if system == "default":
            system = LLAMA2_DEFAULT_SYSTEM_PROMPT
        formatted_instruction = LLAMA2_CAT_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = LLAMA2_CAT_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_llama2_chat(
    tokenizer: AutoTokenizer, instructions: List[str], outputs: List[str] = None, system: str = None, include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_llama2_chat(
                instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama2_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class Llama2CATModel(ModelBase):
    def _load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model.load_adapter(
            peft_model_id="ContinuousAT/Llama-2-7B-CAT",
            device_map="auto",
        )

        model.eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        # From: https://github.com/nrimsky/CAA/blob/main/generate_vectors.py
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer does not have a pad token.")

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_llama2_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(LLAMA2_CAT_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return LLAMA2_CAT_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])
