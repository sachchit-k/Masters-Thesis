"""Smoke test: GPU, RoBERTa fp16, one Captum IG pass, BERT MLM, Sentence-Transformer. No real data."""
import os, sys
for k, v in {"HF_HOME": r"D:\cache\hf", "TORCH_HOME": r"D:\cache\torch"}.items():
    os.environ.setdefault(k, v)
import torch

results = []
def check(name):
    def deco(fn):
        try:
            msg = fn()
            results.append((name, True, msg or ""))
        except Exception as e:  # report, keep going
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco

@check("GPU visible")
def _():
    assert torch.cuda.is_available(), "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, cuda {torch.version.cuda}, free {free/2**30:.2f}/{total/2**30:.2f} GiB"

SENT = "The senator said the bill will cut taxes for everyone."
state = {}

@check("RoBERTa-base fp16 load + forward")
def _():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).half().cuda().eval()
    enc = tok(SENT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits
    state.update(tok=tok, model=model, enc=enc)
    return f"logits shape {tuple(logits.shape)}, dtype {logits.dtype}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"

@check("Captum IG (50 steps, internal_batch_size=5)")
def _():
    from captum.attr import LayerIntegratedGradients
    tok, model, enc = state["tok"], state["model"], state["enc"]
    ids, mask = enc["input_ids"], enc["attention_mask"]
    baseline = torch.full_like(ids, tok.pad_token_id)
    baseline[0, 0], baseline[0, -1] = tok.cls_token_id, tok.sep_token_id
    def fwd(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    lig = LayerIntegratedGradients(fwd, model.roberta.embeddings)
    attr = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask,),
                         target=0, n_steps=50, internal_batch_size=5)
    scores = attr.sum(-1).squeeze(0)
    assert torch.isfinite(scores.float()).all(), "non-finite attributions"
    return f"attr shape {tuple(attr.shape)}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"

# free GPU before the CPU-side models
state.clear(); torch.cuda.empty_cache()

@check("BERT-base MLM load + fill-mask (CPU, no_grad)")
def _():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()
    enc = tok("The senator said the bill will [MASK] taxes.", return_tensors="pt")
    with torch.no_grad():
        out = mlm(**enc).logits
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero().item()
    return f"top-5 candidates computed, logits shape {tuple(out.shape)} (pos {pos})"

@check("Sentence-Transformer load + cosine (CPU, no_grad)")
def _():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    with torch.no_grad():
        e = st.encode([SENT, "The senator claimed the law reduces taxes for all."], convert_to_tensor=True)
    cos = torch.nn.functional.cosine_similarity(e[0], e[1], dim=0).item()
    return f"cosine {cos:.3f}"

@check("spaCy POS tagger (en_core_web_sm)")
def _():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return f"{len(nlp(SENT))} tokens tagged"

print()
for name, ok, msg in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
