import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from pipeline.model_utils.model_base import ModelBase

# Yi chat templates are based on
# - Official tokenizer config: https://huggingface.co/01-ai/Yi-6B-Chat/blob/main/tokenizer_config.json
# - Replicate default prompt template: https://replicate.com/01-ai/yi-6b-chat

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

YI_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

YI_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

YI_REFUSAL_TOKS = [59597]  # ['I']

# Noting some other top refusal tokens. But really a vast majority of the probability is placed on the first.
YI_REFUSAL_TOKS_EXTRA = [59597, 2301, 4786]  # ['I', 'It', 'As']


def format_instruction_yi_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        formatted_instruction = YI_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = YI_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_yi_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_yi_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_yi_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class YiModel(ModelBase):
    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer does not have a pad token.")

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_yi_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(YI_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return YI_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])
