#!/usr/bin/env python3
"""
Comprehensive analysis of ALL places where tokenize_instructions_fn is used
to understand why </s> and trailing space matter.
"""

from transformers import AutoTokenizer

def comprehensive_analysis():
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS OF tokenize_instructions_fn USAGE")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    
    LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    instruction = "How to make a bomb?"
    output = "I cannot help with that request."
    
    # Create both tokenizations
    # MANUAL (instruction only - no output)
    manual_str_no_output = LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)
    manual_tokens_no_output = tokenizer(manual_str_no_output, return_tensors="pt", add_special_tokens=True)
    manual_ids_no_output = manual_tokens_no_output.input_ids[0]
    
    # TEMPLATE (instruction only - no output)
    messages_no_output = [{'role': 'user', 'content': instruction}]
    template_ids_no_output = tokenizer.apply_chat_template(
        messages_no_output, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )[0]
    
    # WITH OUTPUTS
    manual_str_with_output = LLAMA3_CHAT_TEMPLATE.format(instruction=instruction) + output
    manual_tokens_with_output = tokenizer(manual_str_with_output, return_tensors="pt", add_special_tokens=True)
    manual_ids_with_output = manual_tokens_with_output.input_ids[0]
    
    messages_with_output = [
        {'role': 'user', 'content': instruction},
        {'role': 'assistant', 'content': output}
    ]
    template_ids_with_output = tokenizer.apply_chat_template(
        messages_with_output, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )[0]
    
    print("\n" + "=" * 80)
    print("TOKENIZATION COMPARISON (INSTRUCTION ONLY)")
    print("=" * 80)
    
    manual_decoded_no_output = tokenizer.decode(manual_ids_no_output, skip_special_tokens=False)
    template_decoded_no_output = tokenizer.decode(template_ids_no_output, skip_special_tokens=False)
    
    print(f"\nManual: {len(manual_ids_no_output)} tokens")
    print(f"  Decoded: {repr(manual_decoded_no_output)}")
    print(f"  IDs: {manual_ids_no_output.tolist()}")
    
    print(f"\nTemplate: {len(template_ids_no_output)} tokens")
    print(f"  Decoded: {repr(template_decoded_no_output)}")
    print(f"  IDs: {template_ids_no_output.tolist()}")
    
    tokens_identical_no_output = manual_ids_no_output.tolist() == template_ids_no_output.tolist()
    token_diff_no_output = len(manual_ids_no_output) - len(template_ids_no_output)
    
    print(f"\nTokens identical? {tokens_identical_no_output}")
    print(f"Token count difference (manual - template): {token_diff_no_output}")
    
    print("\n" + "=" * 80)
    print("TOKENIZATION COMPARISON (WITH OUTPUT)")
    print("=" * 80)
    
    manual_decoded_with_output = tokenizer.decode(manual_ids_with_output, skip_special_tokens=False)
    template_decoded_with_output = tokenizer.decode(template_ids_with_output, skip_special_tokens=False)
    
    print(f"\nManual: {len(manual_ids_with_output)} tokens")
    print(f"  Decoded: {repr(manual_decoded_with_output)}")
    
    print(f"\nTemplate: {len(template_ids_with_output)} tokens")
    print(f"  Decoded: {repr(template_decoded_with_output)}")
    
    tokens_identical_with_output = manual_ids_with_output.tolist() == template_ids_with_output.tolist()
    token_diff_with_output = len(manual_ids_with_output) - len(template_ids_with_output)
    
    print(f"\nTokens identical? {tokens_identical_with_output}")
    print(f"Token count difference (manual - template): {token_diff_with_output}")
    
    # =========================================================================
    # USE CASE 1: generate_completions
    # =========================================================================
    print("\n" + "=" * 80)
    print("USE CASE 1: generate_completions (model_base.py)")
    print("Uses: tokenize_instructions_fn(instructions=...) - NO outputs")
    print("Critical code: generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]")
    print("=" * 80)
    
    print(f"\nManual input length for slicing: {len(manual_ids_no_output)}")
    print(f"Template input length for slicing: {len(template_ids_no_output)}")
    
    if token_diff_no_output != 0:
        print(f"\nIMPACT: Slice position differs by {abs(token_diff_no_output)} token(s)")
        print("  This means generated tokens would be extracted from different positions!")
    else:
        print("\nNo impact - slice positions are identical.")
    
    # =========================================================================
    # USE CASE 2: get_refusal_scores
    # =========================================================================
    print("\n" + "=" * 80)
    print("USE CASE 2: get_refusal_scores (select_direction.py)")
    print("Uses: tokenize_instructions_fn(instructions=...) - NO outputs")
    print("Critical code: logits[:, -1, :] - uses LAST position's logits")
    print("=" * 80)
    
    manual_last_tok_id = manual_ids_no_output[-1].item()
    template_last_tok_id = template_ids_no_output[-1].item()
    manual_last_tok_str = tokenizer.decode([manual_last_tok_id])
    template_last_tok_str = tokenizer.decode([template_last_tok_id])
    
    print(f"\nManual last token: {manual_last_tok_id} -> {repr(manual_last_tok_str)}")
    print(f"Template last token: {template_last_tok_id} -> {repr(template_last_tok_str)}")
    
    if manual_last_tok_id != template_last_tok_id:
        print(f"\nIMPACT: Different last tokens!")
        print(f"  Manual predicts what comes after {repr(manual_last_tok_str)}")
        print(f"  Template predicts what comes after {repr(template_last_tok_str)}")
        print("  Refusal scores will be computed from different probability distributions!")
    else:
        print("\nNo impact - last tokens are identical.")
    
    # =========================================================================
    # USE CASE 3: get_mean_activations
    # =========================================================================
    print("\n" + "=" * 80)
    print("USE CASE 3: get_mean_activations (generate_directions.py)")
    print("Uses: tokenize_instructions_fn(instructions=...) - NO outputs")
    print("Critical code: activation[:, positions, :] with positions from negative indexing")
    print("=" * 80)
    
    eoi_string = LLAMA3_CHAT_TEMPLATE.split("{instruction}")[-1]
    eoi_toks = tokenizer.encode(eoi_string, add_special_tokens=False)
    n_eoi = len(eoi_toks)
    positions = list(range(-n_eoi, 0))
    
    print(f"\neoi_string: {repr(eoi_string)}")
    print(f"eoi_toks: {eoi_toks}")
    print(f"Number of EOI tokens: {n_eoi}")
    print(f"Positions used for activation extraction: {positions}")
    
    print(f"\nManual - tokens at positions {positions}:")
    manual_at_positions = []
    for pos in positions:
        tok_id = manual_ids_no_output[pos].item()
        tok_str = tokenizer.decode([tok_id])
        manual_at_positions.append((pos, tok_id, tok_str))
        print(f"  pos {pos}: {tok_id} -> {repr(tok_str)}")
    
    print(f"\nTemplate - tokens at positions {positions}:")
    template_at_positions = []
    for pos in positions:
        tok_id = template_ids_no_output[pos].item()
        tok_str = tokenizer.decode([tok_id])
        template_at_positions.append((pos, tok_id, tok_str))
        print(f"  pos {pos}: {tok_id} -> {repr(tok_str)}")
    
    positions_match = all(m[1] == t[1] for m, t in zip(manual_at_positions, template_at_positions))
    
    if not positions_match:
        print("\nIMPACT: Different tokens at extraction positions!")
        print("  Activations will be extracted from different token positions.")
        print("  This completely changes the refusal direction computation!")
        
        # Show mismatches
        print("\n  Mismatched positions:")
        for m, t in zip(manual_at_positions, template_at_positions):
            if m[1] != t[1]:
                print(f"    pos {m[0]}: manual={repr(m[2])} vs template={repr(t[2])}")
    else:
        print("\nNo impact - tokens at all positions are identical.")
    
    # =========================================================================
    # USE CASE 4: batch_iterator_chat_completions (with outputs)
    # =========================================================================
    print("\n" + "=" * 80)
    print("USE CASE 4: batch_iterator_chat_completions (evaluate_loss.py)")
    print("Uses: tokenize_instructions_fn(instructions=..., outputs=...) - WITH outputs")
    print("=" * 80)
    
    print(f"\nManual (with output): {len(manual_ids_with_output)} tokens")
    print(f"  Last 5 tokens: {manual_ids_with_output[-5:].tolist()}")
    print(f"  Last 5 decoded: {[repr(tokenizer.decode([t.item()])) for t in manual_ids_with_output[-5:]]}")
    
    print(f"\nTemplate (with output): {len(template_ids_with_output)} tokens")
    print(f"  Last 5 tokens: {template_ids_with_output[-5:].tolist()}")
    print(f"  Last 5 decoded: {[repr(tokenizer.decode([t.item()])) for t in template_ids_with_output[-5:]]}")
    
    if not tokens_identical_with_output:
        print(f"\nIMPACT: Token sequences differ by {abs(token_diff_with_output)} token(s)")
        
        # Check what's different at the end
        extra_in_template = []
        for i in range(1, abs(token_diff_with_output) + 1):
            if token_diff_with_output < 0:  # template is longer
                tok_id = template_ids_with_output[-i].item()
                extra_in_template.append((tok_id, tokenizer.decode([tok_id])))
        
        if extra_in_template:
            print("  Extra tokens in template (at the end):")
            for tok_id, tok_str in reversed(extra_in_template):
                print(f"    {tok_id} -> {repr(tok_str)}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nInstruction-only case:")
    print(f"  Tokens identical: {tokens_identical_no_output}")
    print(f"  Length difference: {token_diff_no_output}")
    print(f"  Manual ends with: {repr(manual_last_tok_str)}")
    print(f"  Template ends with: {repr(template_last_tok_str)}")
    
    print("\nWith-output case:")
    print(f"  Tokens identical: {tokens_identical_with_output}")
    print(f"  Length difference: {token_diff_with_output}")
    
    print("\nImpacted use cases:")
    impacts = []
    if token_diff_no_output != 0:
        impacts.append("generate_completions (wrong slice position)")
    if manual_last_tok_id != template_last_tok_id:
        impacts.append("get_refusal_scores (different prediction context)")
    if not positions_match:
        impacts.append("get_mean_activations (wrong activation positions)")
    if not tokens_identical_with_output:
        impacts.append("evaluate_loss (different tokens in loss computation)")
    
    if impacts:
        for impact in impacts:
            print(f"  - {impact}")
    else:
        print("  None - tokenizations are equivalent!")
    
    print("\n" + "=" * 80)
    print("CAN apply_chat_template BE USED AS DROP-IN REPLACEMENT?")
    print("=" * 80)
    
    can_replace = tokens_identical_no_output and tokens_identical_with_output
    print(f"\nAnswer: {'YES' if can_replace else 'NO'}")
    
    if not can_replace:
        print("\nReasons:")
        if not tokens_identical_no_output:
            print(f"  - Instruction-only tokenization differs")
            print(f"    Manual: {repr(manual_decoded_no_output)}")
            print(f"    Template: {repr(template_decoded_no_output)}")
        if not tokens_identical_with_output:
            print(f"  - With-output tokenization differs")
            print(f"    Manual: {repr(manual_decoded_with_output)}")
            print(f"    Template: {repr(template_decoded_with_output)}")

if __name__ == "__main__":
    comprehensive_analysis()
