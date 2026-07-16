---
title: Concept Steering Playground
emoji: 🎛️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---

# Concept Steering Playground

Drag the slider to add or subtract a concept direction (formality, optimism,
verbosity) from a small language model's activations at generation time —
no fine-tuning, just an inference-time activation edit. This is a small,
fully-reproducible version of the technique behind demos like Anthropic's
Golden Gate Claude.

Pick a concept and layer in the sidebar, type a prompt, then drag the alpha
slider and hit Generate — negative values push away from the concept,
positive values push toward it, 0 is the unsteered baseline.

Full writeup and source: see the GitHub repo linked from this Space.
