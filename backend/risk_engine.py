"""
Risk Scoring Engine — Antidote AI
Calculates composite risk score and severity from multiple signals.
"""


def calculate_risk(
    poisoning_score: float = 0.0,
    evasion_score: float = 0.0,
    drift_score: float = 0.0,
    model_confidence: float = 1.0,
) -> dict:
    """
    Compute a weighted risk score from four signals.

    Formula
    -------
    risk = 0.25 * poisoning_score
         + 0.25 * evasion_score
         + 0.25 * drift_score
         + 0.25 * (1 - model_confidence) * 100

    All inputs should be in [0, 100] except model_confidence which is [0, 1].

    Parameters
    ----------
    poisoning_score : float
        Poisoning signal (0-100).  100 = data was poisoned.
    evasion_score : float
        Evasion signal (0-100).  Higher = more suspicious.
    drift_score : float
        Drift signal (0-100) from the drift detector.
    model_confidence : float
        Model prediction confidence (0-1).  Lower = riskier.

    Returns
    -------
    dict  {"risk_score": int, "severity": str}
    """
    confidence_penalty = (1.0 - max(0.0, min(1.0, model_confidence))) * 100.0

    raw = (
        0.25 * min(max(poisoning_score, 0), 100)
        + 0.25 * min(max(evasion_score, 0), 100)
        + 0.25 * min(max(drift_score, 0), 100)
        + 0.25 * confidence_penalty
    )

    risk_score = int(min(max(round(raw), 0), 100))

    if risk_score >= 66:
        severity = "HIGH"
    elif risk_score >= 33:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "risk_score": risk_score,
        "severity": severity,
    }
