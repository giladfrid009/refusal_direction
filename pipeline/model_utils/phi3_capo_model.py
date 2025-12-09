import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from pipeline.model_utils.model_base import ModelBase


PHI3_CAPO_CHAT_TEMPLATE = """<|user|> {instruction}<|end|><|assistant|> """

PHI3_CAPO_CHAT_TEMPLATE_WITH_SYSTEM = """<|system|> {system_prompt}<|end|><|user|> {instruction}<|end|><|assistant|> """

PHI3_CAPO_REFUSAL_TOKS = [306]  # [I]


def format_instruction_llama32_chat(instruction: str, output: str = None, system: str = None, include_trailing_whitespace: bool = True):
    if system is not None:
        formatted_instruction = PHI3_CAPO_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = PHI3_CAPO_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_llama3_chat(
    tokenizer: AutoTokenizer, instructions: List[str], outputs: List[str] = None, system: str = None, include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_llama32_chat(
                instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama32_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class Phi3CAPOModel(ModelBase):
    def _load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model.load_adapter(
            peft_model_id="ContinuousAT/Phi-CAPO",
            device_map="auto",
        )

        model.eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer does not have a pad token.")

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_llama3_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(PHI3_CAPO_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return PHI3_CAPO_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])
