"""Two probe backends with one interface for the confirmatory scheduler study.

Both recompute the full sequence on every call (no KV cache), return the greedy token at the
last position, and capture or replace attention-head outputs before the output projection at
one sequence position. A replaced position keeps its donor values on later recomputations as
long as the caller passes the same position.
"""
import ctypes as ct
from pathlib import Path

import numpy as np

INSTRUCTION = "Answer the question using only the exact answer from the passage.\n"
CONTEXT = 256          # native context size used by every earlier QA timing study
MAX_PROMPT = 236       # prompt cap leaving room for the generation cap
MAX_NEW = 14           # greedy generation cap

# Instruction wordings for the chat-template port, compared on the development split only.
STYLES = {
    "v0": dict(system=None, user=lambda c, q: INSTRUCTION + c + "\n" + q),
    "v1": dict(system=None, user=lambda c, q: "Answer the question with only the exact phrase from "
               "the passage. Do not write a sentence.\n" + c + "\n" + q),
    "v2": dict(system="You answer questions with only a short phrase copied exactly from the "
               "passage, never a full sentence.", user=lambda c, q: INSTRUCTION + c + "\n" + q),
    "v3": dict(system=None, user=lambda c, q: c + "\nQuestion: " + q +
               "\nAnswer with only the exact phrase from the passage, not a sentence."),
}


class NativeQwen:
    """Qwen3.5 GGUF through llama.cpp b12110 and qa_confirm_probe.cpp."""
    backend = "llama.cpp-native"

    def __init__(self, library, model, vocab_only=False):
        self.lib = ct.CDLL(str(Path(library).resolve()))
        arr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        mask = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        lib = self.lib
        lib.initialize.argtypes = [ct.c_char_p, ct.c_int]
        lib.tokenize.argtypes = [ct.c_char_p, arr, ct.c_int]
        lib.probe_forward.argtypes = [arr, ct.c_int, mask, ct.c_int, ct.c_int, ct.c_uint32]
        lib.read_buffer.argtypes = [ct.c_int, ct.c_int, f32, ct.c_int]
        lib.write_buffer.argtypes = [ct.c_int, f32, ct.c_int]
        lib.last_logits.restype = ct.POINTER(ct.c_float)
        lib.detokenize.argtypes = [arr, ct.c_int, ct.c_char_p, ct.c_int]
        lib.is_eog.argtypes = [ct.c_int]
        lib.is_eog.restype = ct.c_bool
        self.width = lib.initialize(str(model).encode(), int(vocab_only))
        self.layers, self.heads, self.head_dim = 32, 16, 256
        # Every head of the eight gated full-attention blocks; the 24 Gated DeltaNet blocks
        # have no softmax heads and are never patched.
        self.full_attention = tuple(range(3, 32, 4))
        self.ALL = tuple((layer, head) for layer in self.full_attention for head in range(16))
        self.name = Path(model).name

    def prompt(self, context, question):
        # Qwen3.5 chat template with thinking disabled (identical to qa_subset.prompt).
        return ("<|im_start|>user\n" + INSTRUCTION + context + "\n" + question +
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def ids(self, text):
        buf = np.empty(len(text.encode()) + 1, np.int32)
        n = self.lib.tokenize(text.encode(), buf, len(buf))
        if n <= 0:
            raise ValueError("tokenizer failed")
        return buf[:n].copy()

    def forward(self, ids, sites=(), capture=False, position=None, out_layers=()):
        mask = np.zeros(32, np.uint16)
        for layer, head in sites:
            mask[layer] |= 1 << head
        pos = len(ids) - 1 if position is None else position
        out = sum(1 << layer for layer in out_layers)
        return self.lib.probe_forward(np.ascontiguousarray(ids, np.int32), len(ids), mask,
                                      int(capture), pos, out)

    def distribution(self):
        n = self.lib.vocabulary_size()
        logits = np.ctypeslib.as_array(self.lib.last_logits(), (n,)).astype(np.float64)
        z = logits - logits.max()
        return z - np.log(np.exp(z).sum())

    def is_eog(self, token):
        return bool(self.lib.is_eog(int(token)))

    def text(self, tokens):
        out = ct.create_string_buffer(1024)
        n = self.lib.detokenize(np.asarray(tokens, np.int32), len(tokens), out, len(out))
        if n < 0:
            raise ValueError("detokenization buffer exceeded")
        return out.raw[:n].decode(errors="replace")

    def read(self, layer, kind):
        size = self.heads * self.head_dim if kind == 0 else self.width
        buf = np.empty(size, np.float32)
        n = self.lib.read_buffer(layer, kind, buf, size)
        return buf[:n].copy()

    def write(self, layer, vector):
        vector = np.ascontiguousarray(vector, np.float32)
        self.lib.write_buffer(layer, vector, len(vector))

    def describe(self):
        return dict(backend=self.backend, checkpoint=self.name, blocks=32,
                    patched_blocks=list(self.full_attention), heads_per_block=16, head_dim=256,
                    site="attn_gated-L: sigmoid-gated softmax-attention output per head, input "
                         "of attn_output (wo)", weights="GGUF Q4_K_M", activations="float32",
                    panel_size=len(self.ALL))


class HFModel:
    """Hugging Face causal LM with hooks on the input of every self_attn.o_proj."""
    backend = "transformers"

    def __init__(self, path, dtype="bfloat16", device="cuda", style="v0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.path = Path(path)
        self.name = self.path.name
        self.dtype = getattr(torch, dtype)
        self.style = style
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=self.dtype, attn_implementation="sdpa").to(device).eval()
        cfg = self.model.config
        self.layers = cfg.num_hidden_layers
        self.heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.heads
        self.width = cfg.hidden_size
        self.ALL = tuple((layer, head) for layer in range(self.layers) for head in range(self.heads))
        ends = {self.tok.eos_token_id}
        gen = getattr(self.model, "generation_config", None)
        if gen is not None and gen.eos_token_id is not None:
            ends |= set(np.atleast_1d(gen.eos_token_id).tolist())
        for special in ("<|endoftext|>", "<|im_end|>"):
            idx = self.tok.convert_tokens_to_ids(special)
            if isinstance(idx, int) and idx != self.tok.unk_token_id:
                ends.add(idx)
        self.eog = frozenset(int(x) for x in ends if x is not None)
        self.buffers = [None] * self.layers
        self.outputs = [None] * self.layers
        self._masks = {}
        self.state = None
        blocks = self.model.model.layers
        for layer, block in enumerate(blocks):
            block.self_attn.o_proj.register_forward_pre_hook(self._pre(layer))
            block.self_attn.o_proj.register_forward_hook(self._post(layer))
        self.last = None
        self.context = CONTEXT

    def _pre(self, layer):
        def hook(module, args):
            st = self.state
            entry = st["mask"].get(layer)
            if entry is None:
                return None
            st["seen"][layer] += 1
            x = args[0]
            pos = st["pos"]
            if st["capture"]:
                self.buffers[layer] = x[0, pos].detach().clone()
                return None
            buf = self.buffers[layer]
            assert buf is not None and buf.shape == x[0, pos].shape
            if entry is True:
                x[0, pos] = buf
            else:
                x[0, pos] = self.torch.where(entry, buf, x[0, pos])
            return (x,)
        return hook

    def _post(self, layer):
        def hook(module, args, output):
            st = self.state
            if layer in st["out"]:
                self.outputs[layer] = output[0, st["pos"]].detach().float().cpu().numpy()
        return hook

    def _layer_masks(self, sites):
        key = tuple(sites)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        per = {}
        for layer, head in key:
            per.setdefault(layer, set()).add(head)
        masks = {}
        for layer, heads in per.items():
            if len(heads) == self.heads:
                masks[layer] = True
            else:
                m = self.torch.zeros(self.heads, self.head_dim, dtype=self.torch.bool)
                m[sorted(heads)] = True
                masks[layer] = m.flatten().to(self.device)
        self._masks[key] = masks
        return masks

    def prompt(self, context, question):
        style = STYLES[getattr(self, "style", "v0")]
        messages = [{"role": "user", "content": style["user"](context, question)}]
        if style["system"]:
            messages.insert(0, {"role": "system", "content": style["system"]})
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def ids(self, text):
        return np.asarray(self.tok(text, add_special_tokens=False)["input_ids"], np.int32)

    def forward(self, ids, sites=(), capture=False, position=None, out_layers=()):
        torch = self.torch
        pos = len(ids) - 1 if position is None else position
        assert 0 < len(ids) <= self.context and 0 <= pos < len(ids)
        masks = self._layer_masks(sites)
        self.state = dict(mask=masks, capture=bool(capture), pos=pos, out=set(out_layers),
                          seen={layer: 0 for layer in masks})
        tensor = torch.from_numpy(np.asarray(ids, np.int64)).to(self.device).view(1, -1)
        with torch.inference_mode():
            logits = self.model(input_ids=tensor, use_cache=False, logits_to_keep=1).logits[0, -1]
        self.last = logits
        token = int(torch.argmax(logits).item())   # synchronizes the CUDA stream
        assert all(v == 1 for v in self.state["seen"].values())
        return token

    def distribution(self):
        logits = self.last.double().cpu().numpy()
        z = logits - logits.max()
        return z - np.log(np.exp(z).sum())

    def is_eog(self, token):
        return int(token) in self.eog

    def text(self, tokens):
        return self.tok.decode([int(t) for t in tokens], skip_special_tokens=True)

    def read(self, layer, kind):
        if kind == 0:
            return self.buffers[layer].float().cpu().numpy()
        return self.outputs[layer]

    def write(self, layer, vector):
        self.buffers[layer] = self.torch.as_tensor(np.asarray(vector), dtype=self.dtype,
                                                   device=self.device)

    def projection(self, layer):
        return self.model.model.layers[layer].self_attn.o_proj.weight.detach().float().cpu().numpy()

    def describe(self):
        cfg = self.model.config
        return dict(backend=self.backend, checkpoint=self.name, architecture=cfg.architectures,
                    blocks=self.layers, patched_blocks=list(range(self.layers)),
                    heads_per_block=self.heads, kv_heads=cfg.num_key_value_heads,
                    head_dim=self.head_dim, site="input of self_attn.o_proj (concatenated "
                    "per-head softmax-attention outputs, no gate)", weights=str(self.dtype),
                    activations=str(self.dtype), attention="sdpa", panel_size=len(self.ALL),
                    prompt_style=self.style, context=self.context)
