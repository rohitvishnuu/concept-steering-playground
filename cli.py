#!/usr/bin/env python3
"""
Concept Steering Playground -- CLI entrypoint.

Usage:
    python cli.py "The weather today is" --concept optimism --layer 6
    python cli.py "I think that" --concept formality --alphas -3 0 3 6
"""
import argparse
import datetime
import json
from pathlib import Path

from steering import DEFAULT_MODEL, SteerableModel, list_concepts


def main():
    parser = argparse.ArgumentParser(description="Steer a small LM's generations along a concept direction.")
    parser.add_argument("prompt", help="Prompt to complete")
    parser.add_argument("--concept", required=True, choices=list_concepts(), help="Which concept direction to use")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--layer", type=int, default=6, help="Which transformer layer to steer (default: 6)")
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[-6, -3, 0, 3, 6],
        help="Steering strengths to sweep over (default: -6 -3 0 3 6). 0 = unsteered baseline.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output", help="Output directory for the report")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    sm = SteerableModel(args.model)
    print(f"Model has {sm.n_layers} transformer blocks. Using layer {args.layer}.")

    print(f"Computing steering vector for concept '{args.concept}'...")
    vector = sm.compute_steering_vector(args.concept, args.layer)

    print("Generating sweep across alphas:", args.alphas)
    results = sm.generate_sweep(args.prompt, vector, args.alphas, args.max_new_tokens, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"sweep_{args.concept}_{ts}.md"

    lines = [
        f"# Steering Sweep: {args.concept}\n\n",
        f"**Model:** {args.model}  \n",
        f"**Layer:** {args.layer} / {sm.n_layers}  \n",
        f"**Prompt:** {args.prompt!r}  \n\n",
        "| alpha | generation |\n|---|---|\n",
    ]
    for alpha in args.alphas:
        text = results[alpha].replace("\n", " ").replace("|", "\\|")
        marker = " (baseline)" if alpha == 0 else ""
        lines.append(f"| {alpha}{marker} | {text} |\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}\n")

    for alpha in args.alphas:
        tag = "BASELINE" if alpha == 0 else f"alpha={alpha}"
        print(f"--- {tag} ---\n{results[alpha]}\n")


if __name__ == "__main__":
    main()
