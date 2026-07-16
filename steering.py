"""
Concept Steering Playground -- core logic.

Method (activation addition / "ActAdd"-style steering, in the spirit of
Turner et al. 2023 and the technique behind Anthropic's "Golden Gate Claude"
demo, at a much smaller scale):

  1. Run a small open-weight model over paired contrastive prompts (e.g.
     "formal" phrasings vs. "casual" phrasings of similar content).
  2. At a chosen layer, take the mean residual-stream activation (at the
     final token) for each group and subtract: direction = mean(positive)
     - mean(negative). This is the "concept vector" for that layer.
  3. At generation time, register a forward hook on that layer that adds
     `alpha * direction` to every token's residual stream activation.
  4. Generate with the hook active and compare to unsteered generation.

No fine-tuning, no gradient updates -- purely an inference-time activation
edit. This is intentionally simple (mean-difference-of-activations) rather
than using a full SAE or a trained probe, so it's easy to run on a laptop
and easy to explain in a portfolio writeup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "gpt2"  # 124M params, runs on CPU in seconds per generation
DATA_PATH = Path(__file__).parent / "data" / "concept_pairs.json"


@dataclass
class SteeringVector:
    concept: str
    layer: int
    vector: torch.Tensor  # shape: (hidden_dim,)


class SteerableModel:
    """Wraps a HF causal LM, adds activation-hook-based steering."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._layers = self._locate_layers()
        self.n_layers = len(self._layers)
        self._hook_handle = None

    # -- architecture plumbing -------------------------------------------------
    def _locate_layers(self):
        """Find the list of transformer blocks regardless of exact model family."""
        m = self.model
        for attr_path in [
            ("transformer", "h"),       # GPT-2 family
            ("model", "layers"),        # Llama / Mistral / Gemma family
            ("gpt_neox", "layers"),     # Pythia / GPT-NeoX family
        ]:
            obj = m
            ok = True
            for attr in attr_path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok:
                return obj
        raise ValueError(
            f"Don't know how to locate transformer blocks for {type(m)}. "
            "Add its attribute path to _locate_layers()."
        )

    # -- activation extraction ---------------------------------------------------
    @torch.no_grad()
    def _final_token_activation(self, prompt: str, layer: int) -> torch.Tensor:
        captured = {}

        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured["act"] = hidden[0, -1, :].detach().clone()

        handle = self._layers[layer].register_forward_hook(hook)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            self.model(**inputs)
        finally:
            handle.remove()
        return captured["act"]

    def compute_steering_vector(self, concept: str, layer: int) -> SteeringVector:
        pairs = json.loads(DATA_PATH.read_text())
        if concept not in pairs:
            raise ValueError(f"Unknown concept '{concept}'. Options: {list(pairs)}")
        pos_prompts = pairs[concept]["positive"]
        neg_prompts = pairs[concept]["negative"]

        pos_acts = torch.stack([self._final_token_activation(p, layer) for p in pos_prompts])
        neg_acts = torch.stack([self._final_token_activation(p, layer) for p in neg_prompts])
        direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        return SteeringVector(concept=concept, layer=layer, vector=direction)

    # -- steered generation -----------------------------------------------------
    def _apply_hook(self, vector: torch.Tensor, layer: int, alpha: float):
        vector = vector.to(self.device)

        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                hidden = out[0]
                hidden = hidden + alpha * vector
                return (hidden,) + out[1:]
            return out + alpha * vector

        self._hook_handle = self._layers[layer].register_forward_hook(hook)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        vector: Optional[SteeringVector] = None,
        alpha: float = 0.0,
        seed: int = 0,
    ) -> str:
        torch.manual_seed(seed)
        if vector is not None and alpha != 0.0:
            self._apply_hook(vector.vector, vector.layer, alpha)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        finally:
            self._remove_hook()

    def generate_sweep(
        self,
        prompt: str,
        vector: SteeringVector,
        alphas: List[float],
        max_new_tokens: int = 40,
        seed: int = 0,
    ) -> Dict[float, str]:
        return {a: self.generate(prompt, max_new_tokens, vector, a, seed) for a in alphas}


def list_concepts() -> List[str]:
    return list(json.loads(DATA_PATH.read_text()).keys())
