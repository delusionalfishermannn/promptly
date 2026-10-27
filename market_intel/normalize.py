"""Signal cleaning, feature extraction, language detection and dedupe.

Every feature is a deterministic, explainable function of the signal text plus its
metadata — no black boxes at this stage, because these features feed the opportunity
scores that decide where money is spent.

Feature definitions (all in [0,1] unless noted)
-----------------------------------------------
intent_strength : explicit buying/asking intent ("how do I", "template for",
                  "need a prompt for") + question form + specificity
buyer_segment   : lexicon-weighted argmax over creator/smb/agency/enterprise/
                  educator/student/developer; confidence = margin between top two
urgency         : time-pressure markers ("asap", "deadline", "today")
complexity      : length, clause count, multi-step markers, specialist vocabulary
sentiment       : −1..1 polarity from a small lexicon (negated phrases handled)
pain            : frustration markers ("struggling", "waste of time", "hate")
volume          : provided weight (search volume/impressions/score) — default 1
recency_decay   : exp(-age_days / 14)  (≈10-day half-life)
quality         : penalises boilerplate/too-short/duplicated text
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.clock import utcnow
from core.constants import BUYER_SEGMENTS, SignalKind
from core.ids import content_hash

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")
SENTENCE_SPLIT = re.compile(r"[.!?\n]+")

INTENT_PATTERNS: list[tuple[str, float]] = [
    (r"\bhow (?:do|can|should) (?:i|we|you)\b", 0.55),
    (r"\bhow to\b", 0.45),
    (r"\btemplate for\b", 0.8),
    (r"\bneed (?:a|an|some)? ?(?:prompt|template|swipe|script)\b", 0.85),
    (r"\blooking for (?:a|an)?\b", 0.55),
    (r"\bbest (?:way|tool|practice) to\b", 0.5),
    (r"\bcan (?:someone|anyone)\b", 0.45),
    (r"\brecommend\b", 0.4),
    (r"\bwhich (?:tool|model|prompt)\b", 0.45),
    (r"\bwrite (?:me|a|an)\b", 0.5),
    (r"\bgenerate\b", 0.4),
    (r"\bdraft(?:ing)?\b", 0.35),
    (r"\bsummaris|summariz", 0.3),
    (r"\bfor my (?:business|clients|team|shop|store)\b", 0.5),
    (r"\bwilling to pay|budget|price|cost\b", 0.35),
]

SEGMENT_LEXICON: dict[str, dict[str, float]] = {
    "creator": {
        "youtube": 2,
        "tiktok": 2,
        "instagram": 2,
        "newsletter": 2,
        "substack": 2,
        "creator": 2,
        "influencer": 2,
        "podcast": 2,
        "twitch": 2,
        "reels": 1.5,
        "shorts": 1.5,
        "follower": 1,
        "content calendar": 2,
    },
    "smb": {
        "small business": 3,
        "my business": 2.5,
        "shopify": 2,
        "etsy": 2,
        "invoice": 2,
        "bookkeeping": 2,
        "sole trader": 3,
        "cafe": 2,
        "restaurant": 1.5,
        "salon": 1.5,
        "tradie": 2,
        "customer": 1,
        "quote": 1.5,
        "local": 1,
    },
    "agency": {
        "client": 2,
        "clients": 2,
        "retainer": 2.5,
        "agency": 3,
        "campaign": 2,
        "white label": 3,
        "our clients": 2.5,
        "deliverables": 1.5,
        "account manager": 2,
    },
    "enterprise": {
        "compliance": 2.5,
        "procurement": 2.5,
        "soc 2": 3,
        "enterprise": 3,
        "governance": 2,
        "stakeholder": 2,
        "hr policy": 2,
        "internal policy": 2,
        "team of": 2,
        "department": 1.5,
        "audit": 1.5,
    },
    "educator": {
        "student": 2,
        "students": 2,
        "lesson": 2.5,
        "curriculum": 3,
        "classroom": 3,
        "teacher": 2.5,
        "rubric": 2,
        "assessment": 1.5,
        "course": 2,
    },
    "student": {
        "assignment": 2.5,
        "thesis": 3,
        "uni": 2,
        "university": 2,
        "exam": 2,
        "study": 1.5,
        "essay": 2,
        "dissertation": 3,
    },
    "developer": {
        "api": 2,
        "python": 2,
        "javascript": 2,
        "repo": 2,
        "code": 1.5,
        "sql": 2,
        "docker": 2,
        "backend": 1.5,
        "unit test": 2,
        "regex": 2,
        "json": 1.5,
    },
}

URGENCY_PATTERNS = [
    (r"\basap\b", 0.8),
    (r"\burgent(?:ly)?\b", 0.9),
    (r"\btoday\b", 0.6),
    (r"\bdeadline\b", 0.7),
    (r"\bthis week\b", 0.5),
    (r"\bimmediately\b", 0.8),
    (r"\bnow\b", 0.25),
    (r"!", 0.15),
    (r"\bquick(?:ly)?\b", 0.4),
]

COMPLEXITY_MARKERS = [
    (r"\bmulti[- ]step\b", 0.6),
    (r"\bintegrat", 0.4),
    (r"\bautomate\b", 0.4),
    (r"\bworkflow\b", 0.4),
    (r"\bpipeline\b", 0.35),
    (r"\bdataset\b", 0.3),
    (r"\bapi\b", 0.25),
    (r"\bcomplian", 0.5),
    (r"\bregulat", 0.5),
    (r"\bfor (?:each|every)\b", 0.2),
    (r"\bthen\b.*\bthen\b", 0.3),
]

NEGATIVE_WORDS = {
    "frustrated": 1.0,
    "frustrating": 1.0,
    "hate": 1.0,
    "struggling": 1.2,
    "struggle": 1.2,
    "waste": 1.0,
    "wasting": 1.0,
    "painful": 1.1,
    "annoying": 0.8,
    "confusing": 0.7,
    "broken": 1.0,
    "slow": 0.5,
    "expensive": 0.6,
    "overwhelmed": 1.1,
    "stuck": 0.9,
    "hard": 0.4,
    "difficult": 0.5,
    "messy": 0.6,
    "chaos": 0.8,
    "manual": 0.3,
    "tedious": 0.9,
    "nightmare": 1.2,
    "sucks": 1.0,
    "ugh": 0.7,
    "bad": 0.4,
}
POSITIVE_WORDS = {
    "great": 1.0,
    "love": 1.0,
    "excellent": 1.0,
    "helpful": 0.8,
    "easy": 0.7,
    "perfect": 0.9,
    "awesome": 0.8,
    "thanks": 0.4,
    "amazing": 1.0,
    "works": 0.5,
}
BOILERPLATE = {"thanks", "thank you", "hi", "hello", "please help", "any help appreciated"}

# Tiny stopword-based language hints (enough to route non-English rows to review).
LANG_HINTS: dict[str, set[str]] = {
    "es": {"que", "como", "para", "por", "con", "una", "los", "las", "necesito"},
    "fr": {"que", "pour", "avec", "les", "une", "des", "comment", "besoin"},
    "de": {"und", "für", "mit", "nicht", "eine", "wie", "brauche", "bitte"},
    "pt": {"que", "para", "com", "uma", "não", "como", "preciso"},
    "it": {"che", "per", "con", "una", "non", "come", "bisogno"},
}


@dataclass
class NormalizedDraft:
    text: str
    text_hash: str
    lang: str
    kind: str
    features: dict[str, Any] = field(default_factory=dict)
    intent_strength: float = 0.0
    buyer_segment: str = "unknown"
    urgency: float = 0.0
    complexity: float = 0.0
    sentiment: float = 0.0
    pain: float = 0.0
    volume: float = 1.0
    velocity: float = 0.0
    recency_decay: float = 1.0
    quality: float = 1.0
    weight: float = 1.0
    is_synthetic: bool = False
    collected_at: datetime = field(default_factory=utcnow)
    segment_confidence: float = 0.0
    segment_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_hash": self.text_hash,
            "lang": self.lang,
            "kind": self.kind,
            "features": self.features,
            "intent_strength": round(self.intent_strength, 4),
            "buyer_segment": self.buyer_segment,
            "urgency": round(self.urgency, 4),
            "complexity": round(self.complexity, 4),
            "sentiment": round(self.sentiment, 4),
            "pain": round(self.pain, 4),
            "volume": round(self.volume, 4),
            "velocity": round(self.velocity, 4),
            "recency_decay": round(self.recency_decay, 4),
            "quality": round(self.quality, 4),
            "weight": round(self.weight, 4),
            "is_synthetic": self.is_synthetic,
            "collected_at": self.collected_at.isoformat(),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # strip stray markup
    text = re.sub(r"https?://\S+", " [url] ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    lowered = text.lower()
    tokens = set(WORD_RE.findall(lowered))
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    scores = {lang: len(tokens & hints) for lang, hints in LANG_HINTS.items()}
    best_lang, best = max(scores.items(), key=lambda kv: kv[1])
    english_markers = {"the", "and", "how", "for", "with", "you", "my", "to", "a", "is"}
    if best >= 2 and len(tokens & english_markers) < best:
        return best_lang
    return "en"


def intent_strength(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    hits: list[str] = []
    for pattern, weight in INTENT_PATTERNS:
        if re.search(pattern, lowered):
            score += weight
            hits.append(pattern.strip("\\b"))
    if "?" in text:
        score += 0.2
    words = WORD_RE.findall(lowered)
    if 6 <= len(words) <= 60:
        score += 0.15
    return _clamp(score)


def infer_segment(text: str) -> tuple[str, float, dict[str, float]]:
    lowered = text.lower()
    scores = {segment: 0.0 for segment in BUYER_SEGMENTS if segment != "unknown"}
    for segment, lexicon in SEGMENT_LEXICON.items():
        for phrase, weight in lexicon.items():
            if phrase in lowered:
                scores[segment] += weight
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_segment, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < 1.5:
        return "unknown", 0.2, scores
    margin = (top_score - second_score) / max(top_score, 1e-9)
    confidence = _clamp(0.45 + 0.55 * margin)
    return top_segment, confidence, scores


def urgency_score(text: str) -> float:
    lowered = text.lower()
    score = sum(weight for pattern, weight in URGENCY_PATTERNS if re.search(pattern, lowered))
    return _clamp(score)


def complexity_score(text: str) -> float:
    lowered = text.lower()
    words = WORD_RE.findall(lowered)
    base = min(len(words) / 120.0, 0.4)
    markers = sum(weight for pattern, weight in COMPLEXITY_MARKERS if re.search(pattern, lowered))
    clauses = len(SENTENCE_SPLIT.split(text))
    clause_factor = min(clauses / 12.0, 0.25)
    return _clamp(base + markers + clause_factor)


def sentiment_and_pain(text: str) -> tuple[float, float]:
    lowered = text.lower()
    tokens = WORD_RE.findall(lowered)
    neg = sum(NEGATIVE_WORDS.get(tok, 0.0) for tok in tokens)
    # Handle simple negation ("not difficult")
    if re.search(r"\bnot (?:very )?(?:hard|difficult|bad)\b", lowered):
        neg = max(0.0, neg - 0.5)
    pos = sum(POSITIVE_WORDS.get(tok, 0.0) for tok in tokens)
    total = neg + pos
    sentiment = 0.0 if total == 0 else (pos - neg) / max(total, 1e-9)
    pain = _clamp(neg / 2.5)
    return round(sentiment, 4), pain


def quality_score(text: str, words: int) -> float:
    lowered = text.strip().lower()
    if lowered in BOILERPLATE:
        return 0.1
    quality = 1.0
    if words < 5:
        quality -= 0.5
    elif words < 9:
        quality -= 0.2
    if len(set(WORD_RE.findall(lowered))) <= max(2, words // 4):
        quality -= 0.25  # repetitive/keyword-stuffed
    if re.search(r"(.)\1{5,}", lowered):
        quality -= 0.2
    if re.search(r"\b(viagra|casino|crypto giveaway|forex signals)\b", lowered):
        quality -= 0.6  # spam / low-value demand
    return _clamp(quality)


def recency_decay(
    occurred_at: datetime, *, now: datetime | None = None, half_life_days: float = 10.0
) -> float:
    now = now or utcnow()
    reference = occurred_at if occurred_at.tzinfo else occurred_at.replace(tzinfo=now.tzinfo)
    age_days = max((now - reference).total_seconds() / 86400.0, 0.0)
    return float(math.exp(-age_days / (half_life_days / math.log(2))))


def infer_kind(text: str, declared: str | None) -> str:
    if declared and declared != str(SignalKind.OTHER):
        return declared
    lowered = text.lower()
    if "?" in text or re.match(r"^(how|what|why|when|where|which|can|should)\b", lowered):
        return str(SignalKind.QUESTION)
    if re.search(r"\b(bug|error|broken|not working|crashed)\b", lowered):
        return str(SignalKind.SUPPORT_TICKET)
    return str(SignalKind.OTHER)


def normalize_text(
    text: str,
    *,
    declared_kind: str | None = None,
    occurred_at: datetime | None = None,
    volume: float = 1.0,
    is_synthetic: bool = False,
    now: datetime | None = None,
    segment_hint: str | None = None,
) -> NormalizedDraft:
    """Turn one raw text into a fully-featured draft (no DB, no network)."""
    now = now or utcnow()
    cleaned = clean_text(text)
    words = WORD_RE.findall(cleaned)
    intent = intent_strength(cleaned)
    segment, segment_conf, segment_scores = infer_segment(cleaned)
    if segment_hint and segment_hint in BUYER_SEGMENTS:
        segment, segment_conf = segment_hint, max(segment_conf, 0.9)
    sentiment, pain = sentiment_and_pain(cleaned)
    urgency = urgency_score(cleaned)
    complexity = complexity_score(cleaned)
    quality = quality_score(cleaned, len(words))
    decay = recency_decay(occurred_at or now, now=now)
    lang = detect_language(cleaned)

    weight = _clamp(
        decay * quality * (0.5 + 0.5 * intent) * math.log1p(max(volume, 0.0)) / math.log(11),
        0.0,
        5.0,
    )
    features = {
        "words": len(words),
        "has_question": "?" in cleaned,
        "segment_scores": {k: round(v, 2) for k, v in segment_scores.items() if v > 0},
        "segment_confidence": round(segment_conf, 3),
        "intent_hits": intent,
        "lang": lang,
        "decay": round(decay, 4),
    }
    return NormalizedDraft(
        text=cleaned,
        text_hash=content_hash(cleaned.lower()),
        lang=lang,
        kind=infer_kind(cleaned, declared_kind),
        features=features,
        intent_strength=intent,
        buyer_segment=segment,
        urgency=urgency,
        complexity=complexity,
        sentiment=sentiment,
        pain=pain,
        volume=float(volume),
        velocity=0.0,
        recency_decay=decay,
        quality=quality,
        weight=weight,
        is_synthetic=is_synthetic,
        collected_at=occurred_at or now,
        segment_confidence=segment_conf,
        segment_scores=segment_scores,
    )


def shingles(text: str, size: int = 5) -> set[str]:
    tokens = WORD_RE.findall(text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
