"""
Ensemble Decision — Antidote AI
Combines poisoning, evasion, drift, and model prediction into a final verdict.
"""


def ensemble_decision(
    poisoning_flag: bool,
    evasion_flag: bool,
    model_prediction: int,
    evasion_score: float = 0.0,
    model_confidence: float = 1.0,
    drift_flag: bool = False,
    severity: str = "LOW",
    explanation: list | None = None,
) -> dict:
    """
    Produce the final security decision.

    Decision logic
    --------------
    - poisoning_flag             → BLOCK  (data was tampered with)
    - evasion_flag AND drift_flag → BLOCK  (adversarial + distribution shift)
    - evasion_flag               → FLAG   (input looks adversarial)
    - drift_flag + severity HIGH → FLAG   (significant distribution shift)
    - model_prediction == 1      → BLOCK  (model says malicious)
    - otherwise                  → ALLOW

    Returns
    -------
    dict with keys:
        decision   – "BLOCK" | "FLAG" | "ALLOW"
        risk_score – 0-100 integer  (legacy, kept for backward compat)
        details    – human-readable explanation
    """

    explanation = explanation or []

    # ── Decision ──────────────────────────────────────────
    if poisoning_flag:
        decision = "BLOCK"
        details = "Poisoning detected in dataset — input blocked."
    elif evasion_flag and drift_flag:
        decision = "BLOCK"
        details = "Evasion attempt combined with distribution drift — input blocked."
    elif evasion_flag:
        decision = "FLAG"
        details = "Evasion attempt detected — input flagged for review."
    elif drift_flag and severity == "HIGH":
        decision = "FLAG"
        details = "Significant distribution drift detected — input flagged for review."
    elif model_prediction == 1:
        decision = "BLOCK"
        details = "Model predicts malicious class — input blocked."
    else:
        decision = "ALLOW"
        details = "All checks passed — input allowed."

    # Append drift status
    if drift_flag:
        details += " [Drift: DETECTED]"
    else:
        details += " [Drift: CLEAR]"

    # Append explanation
    if explanation and explanation[0] != "All features within expected range":
        details += " | Explanation: " + "; ".join(explanation)

    # ── Risk score (legacy — kept for backward compat) ────
    risk = 0.0
    if poisoning_flag:
        risk += 45
    if evasion_flag:
        risk += 30
    if drift_flag:
        risk += 15
    if model_prediction == 1:
        risk += 25

    risk += abs(evasion_score) * 10
    risk += (1.0 - model_confidence) * 15

    risk_score = int(min(max(risk, 0), 100))

    return {
        "decision": decision,
        "risk_score": risk_score,
        "details": details,
    }
