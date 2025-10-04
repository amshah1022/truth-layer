import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfFolder
import openai 

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

# Get token from env var or from HF CLI login (~/.huggingface)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()

openai.api_key = os.getenv("OPENAI_API_KEY")
def _auth_kwargs():
    import os
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return {"token": tok}  
    return {}


_tokenizer = _model = _gen = None

def _load_hf(model_id: str):
    global _tokenizer, _model, _gen
    if _gen is not None and getattr(_gen, "_model_id", None) == model_id:
        return _gen

    if "meta-llama/" in model_id and not HF_TOKEN:
        raise RuntimeError(
            "Llama-3 is gated and no HF token was found. "
            "Export HUGGINGFACE_HUB_TOKEN=hf_xxx or run `huggingface-cli login`."
        )

    auth = _auth_kwargs()
    print(f"[models] Loading {model_id} | token={'yes' if HF_TOKEN else 'no'}")

    _tokenizer = AutoTokenizer.from_pretrained(model_id, **auth)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        **auth,
    )
    _gen = pipeline("text-generation", model=_model, tokenizer=_tokenizer, return_full_text=False)
    _gen._model_id = model_id
    return _gen

def _build_prompt(q: str) -> str:
    return (
        "You are a concise QA model. "
        "Answer with ONLY the minimal text span (no punctuation, no extra words). "
        "If unsure, answer exactly: Unknown.\n"
        f"Q: {q}\nA:"
    )

def generate_answer(question: str, max_new_tokens: int = 16) -> str:
    # OpenAI models
    if MODEL_ID.startswith("gpt-"):
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Export it first.")
        prompt = _build_prompt(question)
        resp = openai.chat.completions.create(
            model=MODEL_ID,  # e.g., "gpt-4o-mini" or "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0
        )
        ans = resp.choices[0].message.content.strip()
        ans = ans.strip(" .,:;!?\"'()[]{}")
        return ans

    # Hugging Face models 
    gen = _load_hf(MODEL_ID)
    out = gen(
        _build_prompt(question),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=gen.tokenizer.pad_token_id,
    )[0]["generated_text"]
    ans = out.splitlines()[0]
    ans = ans.replace("Answer:", "").replace("A:", "").strip()
    ans = ans.strip(" .,:;!?\"'()[]{}")
    return ans




