#mock.py
import random
from typing import List, Dict

# A tiny scriptable set so can demo a clear lie -> fix
WRONG_BASELINES = {
    "Who wrote Pride and Prejudice?": "Pride and Prejudice was written by Charlotte Brontë.",
    "When was Cornell University founded?": "Cornell University was founded in 1965.",
    "What is the capital of Australia?": "Sydney is the capital of Australia."
}

def mock_baseline_answer(question: str) -> str:
    # If we have a scripted wrong answer, use it; else give a generic plausible answer
    if question in WRONG_BASELINES:
        return WRONG_BASELINES[question]
    # light randomness so it feels alive (never 100% same)
    fillers = [
        "From what I recall, it might be {}.",
        "It’s commonly thought to be {}.",
        "I believe the answer is {}."
    ]
    guesses = ["42", "unknown", "not clearly documented", "N/A"]
    return random.choice(fillers).format(random.choice(guesses))

def mock_mitigated_answer(question: str, sources: List[Dict]) -> str:
    """
    Build a grounded, cited answer using snippets. We just stitch a short
    summary from the first snippet and cite [S1].
    """
    if not sources:
        return "Insufficient evidence in the provided sources."
    s1 = sources[0]
    # Keep it short; pretend we extracted a relevant fact
    text = s1.get("text", "")
    # Take a small slice
    summary = text[:220].rsplit(" ", 1)[0]
    return f"{summary} [S1]"
