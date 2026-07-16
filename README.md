# Concept Steering Playground

An interactive tool for **activation steering**: nudging a language model's
internal activations at generation time to shift its output along a chosen
concept (formality, optimism, verbosity) — without any fine-tuning, just a
vector added to the residual stream mid-inference. Drag a slider, watch the
same prompt's completion shift in real time.

## Motivation

Anthropic's ["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude)
demo showed that amplifying a single interpretable feature inside a model
could make it fixate on a concept, live, with no retraining. The broader
technique — finding a direction in activation space that corresponds to a
concept, then adding or subtracting it during generation — goes back to
activation-engineering work like Turner et al.'s "Activation Addition" (2023)
and the "steering vectors" literature more generally.

This project builds a small, fully-reproducible version of that technique on
an open-weight model, with a live slider UI so the effect is something you
can *see happen*, not just read about. No API key, no cost — it runs
entirely on CPU with a small local model.

## How it works

```
"formal" example sentences  ─┐
                              ├─► mean activation (layer L) ─┐
"casual" example sentences  ─┘                                ├─► direction = formal - casual
                              ├─► mean activation (layer L) ─┘
```

1. Pick a concept (e.g. `formality`) and a layer `L`.
2. Run a handful of hand-written contrastive sentence pairs (formal vs.
   casual phrasings of similar content) through the model, and record the
   residual-stream activation at layer `L` for each.
3. `direction = mean(formal activations) - mean(casual activations)` — a
   single vector in activation space that (approximately) encodes "more
   formal."
4. At generation time, register a forward hook on layer `L` that adds
   `alpha * direction` to every token's activation as it passes through.
   `alpha > 0` pushes toward the concept, `alpha < 0` pushes away from it,
   `alpha = 0` is the unsteered baseline.

This is deliberately the simplest version of the technique (mean-difference
of activations, not a trained probe or sparse autoencoder feature) so it's
easy to run on a laptop and easy to explain end-to-end in an interview or
application essay.

## Setup

```bash
git clone <this-repo>
cd concept-steering-playground
pip install -r requirements.txt
```

No API key needed. Default model is `gpt2` (124M params) so the first run
just downloads a small model from Hugging Face and everything after that is
CPU-fast.

## Usage

**Interactive slider demo (the fun part):**

```bash
streamlit run app.py
```

Pick a concept and layer in the sidebar, type a prompt, then drag the alpha
slider and hit Generate. This is the piece worth recording a short screen
capture of for a README GIF / LinkedIn post — the live tonal shift is the
whole point of this project.

**CLI sweep (for reproducible output you can commit to the repo):**

```bash
python cli.py "I think that the future of technology" --concept optimism --layer 6
```

This generates completions across a range of alpha values and writes a
markdown table to `output/sweep_<concept>_<timestamp>.md`. Run this once for
each concept and commit the reports — a real sweep table is much more
convincing on a portfolio than a description of one.

## Testing

```bash
python -m pytest test_steering.py -v
```

These tests check the contrastive-pair dataset (balance, no duplicates, no
empty examples) without needing to load a model — fast and free to run in
CI.

## Choosing a layer

Middle layers (roughly layers 4–8 out of GPT-2's 12) tend to carry the most
semantically abstract, steerable representations — very early layers are
closer to raw token identity, very late layers are closer to next-token
logits. The `--layer` / sidebar slider lets you sweep this yourself; part of
the value of this project is seeing *empirically* which layer steers best
for a given concept, rather than taking that on faith.

## Limitations

- Concept directions are derived from a small, hand-written set of
  contrastive sentences (4–8 pairs per concept) — a real research version
  would validate on a held-out set and report a quantitative "steering
  success rate," not just qualitative examples.
- Mean-difference-of-activations is a blunt instrument compared to a trained
  linear probe or an SAE feature — it can pick up correlated concepts
  alongside the target one (e.g. "formality" may partially entangle with
  "sentence length").
- Only tested on GPT-2-scale models here; larger models may need different
  layer choices or steering magnitudes.
- No automatic detection of when steering has broken the model's fluency
  entirely (very high `alpha` values can produce degenerate text) — this is
  visible qualitatively in the slider demo but not measured numerically.

## Possible extensions

- Add a fluency/perplexity check alongside each steered generation, so you
  can quantify the fluency-vs-steering-strength tradeoff, not just eyeball it.
- Compare mean-difference vectors against a proper linear probe trained to
  classify formal vs. casual text, and see how much the two directions agree.
- Extend to a mid-size open model (e.g. Llama-3-8B) on a GPU and see whether
  steering directions transfer across model scales.

## License

MIT — see [LICENSE](LICENSE).
