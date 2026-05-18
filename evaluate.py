"""Evaluation for MDLM Multi-Head Distillation.

Implements:
- Generative PPL (using GPT-2 Large as scorer)
- MAUVE score
- T_outer sweep diagnostics
"""

import os
import json
import torch
import torch.nn.functional as F
import transformers
import numpy as np

from config import Config
from model import build_student, build_teacher
from inference import generate_samples, generate_samples_baseline_mdlm


@torch.no_grad()
def compute_generative_ppl(
    text_samples: list,
    eval_model_name: str = "gpt2-large",
    max_length: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
) -> float:
    """
    Compute generative perplexity using an AR model (GPT-2 Large).

    Following SDTT/MDLM evaluation protocol.

    Args:
        text_samples: list of generated text strings
        eval_model_name: name of the eval AR model
        max_length: max sequence length for eval model
        batch_size: batch size for eval
        device: device

    Returns:
        perplexity: generative PPL
    """
    eval_tokenizer = transformers.AutoTokenizer.from_pretrained(eval_model_name)
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id

    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        eval_model_name).eval().to(device)

    # Tokenize
    encodings = eval_tokenizer(
        text_samples,
        return_tensors='pt',
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    input_ids = encodings['input_ids'].to(device)
    attn_mask = encodings['attention_mask'].to(device)

    total_nll = 0.0
    total_tokens = 0

    n_batches = (len(text_samples) + batch_size - 1) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(text_samples))
        batch_ids = input_ids[start:end]
        batch_mask = attn_mask[start:end]

        # Split into context-sized chunks
        for chunk_start in range(0, batch_ids.shape[1], max_length):
            chunk_ids = batch_ids[:, chunk_start:chunk_start + max_length]
            chunk_mask = batch_mask[:, chunk_start:chunk_start + max_length]

            if chunk_ids.shape[1] < 2:
                continue

            logits = eval_model(chunk_ids, attention_mask=chunk_mask).logits
            # NLL: cross-entropy of predicting next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()
            shift_mask = chunk_mask[:, 1:].contiguous()

            nlls = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
            ).view(shift_labels.shape)

            # Mask out padding and tokens after first EOS
            first_eos = (chunk_ids == eval_tokenizer.eos_token_id).cumsum(-1) == 1
            token_mask = (chunk_ids != eval_tokenizer.eos_token_id)
            valid_mask = (first_eos[:, 1:] + token_mask[:, 1:]).float() * shift_mask.float()

            total_nll += (nlls * valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()

    del eval_model
    torch.cuda.empty_cache()

    if total_tokens == 0:
        return float('inf')

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def compute_mauve(
    generated_texts: list,
    reference_texts: list,
    device_id: int = 0,
    max_text_length: int = 256,
    num_buckets: str = "auto",
) -> float:
    """
    Compute MAUVE score between generated and reference texts.

    Args:
        generated_texts: list of generated text strings
        reference_texts: list of reference text strings
        device_id: GPU device
        max_text_length: max text length for MAUVE

    Returns:
        mauve_score: MAUVE score (0 to 1)
    """
    import mauve

    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
        batch_size=64,
        featurize_model_name="gpt2-large",
    )
    return result.mauve


import math


def evaluate_model(config: Config, checkpoint_path: str = None):
    """
    Full evaluation pipeline.

    Args:
        config: configuration
        checkpoint_path: path to student checkpoint
    """
    device = config.device
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Build and load student
    print("\nBuilding student...")
    student = build_student(config, device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        student.backbone_loras.load_state_dict(ckpt['student_backbone_loras'])
        student.heads.load_state_dict(ckpt['student_heads'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    student.eval()

    # Build teacher
    print("Building teacher (for baseline)...")
    teacher = build_teacher(config, device)

    results = {}
    num_samples = config.num_eval_samples

    # ===== MH-K4 at T_outer=4 (headline) =====
    print(f"\n--- MH-K4 (T_outer={config.T_outer}, K={config.K}) ---")
    samples = generate_samples(student, config, num_samples=num_samples,
                               T_outer=config.T_outer, device=device)
    texts = tokenizer.batch_decode(samples.cpu())
    ppl = compute_generative_ppl(texts, config.gen_ppl_model, device=device)
    print(f"  Generative PPL: {ppl:.2f}")
    results[f'MH-K4_T{config.T_outer}'] = {'ppl': ppl}

    # Save samples
    sample_path = os.path.join(config.output_dir, f"eval_samples_MH-K4_T{config.T_outer}.txt")
    with open(sample_path, 'w') as f:
        for i, text in enumerate(texts[:64]):
            f.write(f"=== Sample {i} ===\n{text}\n\n")

    # ===== T_outer sweep for MH-K4 =====
    for t_outer_test in [1, 2, 8]:
        print(f"\n--- MH-K4 diagnostic (T_outer={t_outer_test}, K={config.K}) ---")
        samples = generate_samples(student, config, num_samples=num_samples,
                                   T_outer=t_outer_test, device=device)
        texts = tokenizer.batch_decode(samples.cpu())
        ppl = compute_generative_ppl(texts, config.gen_ppl_model, device=device)
        print(f"  Generative PPL: {ppl:.2f}")
        results[f'MH-K4_T{t_outer_test}'] = {'ppl': ppl}

    # ===== MDLM baseline =====
    for num_steps in [4, 8, 16, 32, 64]:
        print(f"\n--- MDLM baseline ({num_steps} substeps) ---")
        samples = generate_samples_baseline_mdlm(
            teacher, config, num_samples=num_samples,
            num_steps=num_steps, device=device)
        texts = tokenizer.batch_decode(samples.cpu())
        ppl = compute_generative_ppl(texts, config.gen_ppl_model, device=device)
        print(f"  Generative PPL: {ppl:.2f}")
        results[f'MDLM_{num_steps}'] = {'ppl': ppl}

    # Save results
    results_path = os.path.join(config.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'PPL':>10}")
    print("-" * 35)
    for name, vals in results.items():
        print(f"{name:<25} {vals['ppl']:>10.2f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    config = Config(output_dir=args.output_dir)
    evaluate_model(config, args.checkpoint)
