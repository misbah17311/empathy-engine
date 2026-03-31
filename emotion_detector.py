"""
Emotion Detection Module
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.
Classifies text into granular emotional categories with intensity scores.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()


def detect_emotion(text: str) -> dict:
    """
    Analyze text and return detected emotion with intensity.

    Returns:
        dict with keys:
            - category: str (e.g., "happy", "angry", "sad", "surprised", "concerned", "neutral")
            - intensity: float (0.0 to 1.0, how strong the emotion is)
            - compound: float (-1.0 to 1.0, raw VADER compound score)
            - scores: dict (raw VADER pos/neg/neu/compound scores)
    """
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    category, intensity = _classify_emotion(text, scores)

    return {
        "category": category,
        "intensity": intensity,
        "compound": compound,
        "scores": scores,
    }


def _classify_emotion(text: str, scores: dict) -> tuple:
    """
    Map VADER scores to granular emotion categories.
    Uses compound score, individual pos/neg scores, and text heuristics
    to distinguish between nuanced emotions.
    """
    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    neu = scores["neu"]

    text_lower = text.lower()

    # Strong positive
    if compound >= 0.6:
        # Check for excitement/surprise markers
        if any(w in text_lower for w in ["wow", "amazing", "incredible", "unbelievable", "!", "awesome"]):
            return ("surprised", round(abs(compound), 2))
        return ("happy", round(abs(compound), 2))

    # Moderate positive
    if compound >= 0.2:
        # Check for inquisitive tone
        if "?" in text:
            return ("inquisitive", round(abs(compound), 2))
        return ("happy", round(abs(compound), 2))

    # Slight positive to slight negative — neutral zone
    if -0.2 < compound < 0.2:
        if "?" in text:
            return ("inquisitive", 0.3)
        if any(w in text_lower for w in ["worry", "worried", "concern", "concerned", "hope", "hopefully", "unsure"]):
            return ("concerned", 0.4)
        return ("neutral", round(1.0 - abs(compound), 2))

    # Moderate negative
    if compound <= -0.2 and compound > -0.6:
        if any(w in text_lower for w in ["worry", "worried", "concern", "afraid", "anxious", "nervous"]):
            return ("concerned", round(abs(compound), 2))
        if "?" in text:
            return ("concerned", round(abs(compound), 2))
        return ("sad", round(abs(compound), 2))

    # Strong negative
    if compound <= -0.6:
        if any(w in text_lower for w in ["hate", "angry", "furious", "annoyed", "frustrated", "terrible", "worst"]):
            return ("angry", round(abs(compound), 2))
        return ("sad", round(abs(compound), 2))

    return ("neutral", 0.5)


# Convenience: list all supported emotion categories
EMOTION_CATEGORIES = ["happy", "sad", "angry", "surprised", "inquisitive", "concerned", "neutral"]
