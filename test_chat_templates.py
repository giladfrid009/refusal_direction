#!/usr/bin/env python3
"""
Print chat templates for various models using tokenizer.apply_chat_template
"""

from transformers import AutoTokenizer

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen3-4B-Instruct-2507",
    "microsoft/Phi-3-mini-4k-instruct",
    "cais/zephyr_7b_r2d2",
    "GraySwanAI/Llama-3-8B-Instruct-RR",
    "LLM-LAT/robust-llama3-8b-instruct",
]

INSTRUCTION = "How do I make a bomb?"
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT = "I cannot help with that request."


def test_model_chat_template(model_name: str):
    print("\n" + "=" * 100)
    print(f"MODEL: {model_name}")
    print("=" * 100)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        return
    
    # Case 1: Instruction only (no system prompt)
    print("\n" + "-" * 80)
    print("CASE 1: Instruction only (no system prompt)")
    print("-" * 80)
    
    messages_1 = [
        {"role": "user", "content": INSTRUCTION}
    ]
    
    try:
        tokens_1 = tokenizer.apply_chat_template(
            messages_1,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        decoded_1 = tokenizer.decode(tokens_1[0], skip_special_tokens=False)
        print(f"Tokens: {tokens_1[0].tolist()}")
        print(f"Decoded:\n{repr(decoded_1)}")
        
        # Re-tokenize the decoded string and compare
        retokenized_1 = tokenizer(decoded_1, truncation=False, return_tensors="pt")
        retok_ids_1 = retokenized_1["input_ids"][0].tolist()
        orig_ids_1 = tokens_1[0].tolist()
        if retok_ids_1 == orig_ids_1:
            print("✓ Re-tokenization MATCHES")
        else:
            print("✗ Re-tokenization MISMATCH!")
            print(f"  Original:     {orig_ids_1}")
            print(f"  Re-tokenized: {retok_ids_1}")
            if len(retok_ids_1) > len(orig_ids_1) and retok_ids_1[1:] == orig_ids_1:
                print(f"  -> tokenizer() added extra BOS token: {retok_ids_1[0]}")
            elif len(retok_ids_1) < len(orig_ids_1) and retok_ids_1 == orig_ids_1[1:]:
                print(f"  -> apply_chat_template added extra BOS token: {orig_ids_1[0]}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Case 2: System prompt + Instruction
    print("\n" + "-" * 80)
    print("CASE 2: System prompt + Instruction")
    print("-" * 80)
    
    messages_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION}
    ]
    
    try:
        tokens_2 = tokenizer.apply_chat_template(
            messages_2,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        decoded_2 = tokenizer.decode(tokens_2[0], skip_special_tokens=False)
        print(f"Tokens: {tokens_2[0].tolist()}")
        print(f"Decoded:\n{repr(decoded_2)}")
        
        # Re-tokenize the decoded string and compare
        retokenized_2 = tokenizer(decoded_2, truncation=False, return_tensors="pt")
        retok_ids_2 = retokenized_2["input_ids"][0].tolist()
        orig_ids_2 = tokens_2[0].tolist()
        if retok_ids_2 == orig_ids_2:
            print("✓ Re-tokenization MATCHES")
        else:
            print("✗ Re-tokenization MISMATCH!")
            print(f"  Original:     {orig_ids_2}")
            print(f"  Re-tokenized: {retok_ids_2}")
            if len(retok_ids_2) > len(orig_ids_2) and retok_ids_2[1:] == orig_ids_2:
                print(f"  -> tokenizer() added extra BOS token: {retok_ids_2[0]}")
            elif len(retok_ids_2) < len(orig_ids_2) and retok_ids_2 == orig_ids_2[1:]:
                print(f"  -> apply_chat_template added extra BOS token: {orig_ids_2[0]}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Case 3: System prompt + Instruction + Output
    print("\n" + "-" * 80)
    print("CASE 3: System prompt + Instruction + Output")
    print("-" * 80)
    
    messages_3 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION},
        {"role": "assistant", "content": OUTPUT}
    ]
    
    try:
        tokens_3 = tokenizer.apply_chat_template(
            messages_3,
            tokenize=True,
            add_generation_prompt=False,  # Don't add generation prompt after assistant response
            return_tensors="pt"
        )
        decoded_3 = tokenizer.decode(tokens_3[0], skip_special_tokens=False)
        print(f"Tokens: {tokens_3[0].tolist()}")
        print(f"Decoded:\n{repr(decoded_3)}")
        
        # Re-tokenize the decoded string and compare
        retokenized_3 = tokenizer(decoded_3, truncation=False, return_tensors="pt")
        retok_ids_3 = retokenized_3["input_ids"][0].tolist()
        orig_ids_3 = tokens_3[0].tolist()
        if retok_ids_3 == orig_ids_3:
            print("✓ Re-tokenization MATCHES")
        else:
            print("✗ Re-tokenization MISMATCH!")
            print(f"  Original:     {orig_ids_3}")
            print(f"  Re-tokenized: {retok_ids_3}")
            if len(retok_ids_3) > len(orig_ids_3) and retok_ids_3[1:] == orig_ids_3:
                print(f"  -> tokenizer() added extra BOS token: {retok_ids_3[0]}")
            elif len(retok_ids_3) < len(orig_ids_3) and retok_ids_3 == orig_ids_3[1:]:
                print(f"  -> apply_chat_template added extra BOS token: {orig_ids_3[0]}")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    print("=" * 100)
    print("CHAT TEMPLATE COMPARISON ACROSS MODELS")
    print("=" * 100)
    print(f"\nInstruction: {repr(INSTRUCTION)}")
    print(f"System prompt: {repr(SYSTEM_PROMPT)}")
    print(f"Output: {repr(OUTPUT)}")
    
    for model_name in MODELS:
        test_model_chat_template(model_name)
    
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
