"""IP/originality checks: block near-duplicates of our catalog and any permitted corpus.

Two independent signals, because each fails differently:

1. **Embedding cosine** (semantic) — catches paraphrases and light rewrites.
2. **5-gram shingle Jaccard** (lexical) — catches verbatim/near-verbatim reuse that a
   semantic model might score as merely "similar".

Comparison targets, and *only* these:

* our own catalog (``prompt_variants``) — prevents self-cannibalisation and accidental
  re-release of an existing product;
* a **permitted corpus directory** with a manifest that declares licence and
  permission basis for every file. Anything without a declaration is ignored.
  Competitor prompts are never ingested, so they can never be compared against —
  which structurally prevents "light rewrite" behaviour.

Blocking is deliberate: a blocked product cannot be approved. The operator can revise
the IR (new structure, new examples) and re-run the gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import get_settings
from core.constants import GovernanceDecision, PermissionBasis
from db.models import PromptVariant
from governance.policy import Finding
from llm.embeddings import cosine, get_embedder
from market_intel.normalize import jaccard, shingles

PERMITTED_BASES = {
    str(PermissionBasis.PERMISSIVE_LICENSE),
    str(PermissionBasis.PUBLIC_DOMAIN),
    str(PermissionBasis.SYNTHETIC),
    str(PermissionBasis.OPERATOR_PROVIDED),
}


@dataclass
class CorpusDocument:
    ref: str
    text: str
    license: str
    permission_basis: str


@dataclass
class SimilarityResult:
    decision: str
    max_similarity: float
    shingle_overlap: float
    nearest_ref: str
    threshold: float
    compared_against: str
    n_catalog: int = 0
    n_corpus: int = 0
    findings: list[Finding] = field(default_factory=list)
    skipped_corpus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "max_similarity": round(self.max_similarity, 4),
            "shingle_overlap": round(self.shingle_overlap, 4),
            "nearest_ref": self.nearest_ref,
            "threshold": self.threshold,
            "compared_against": self.compared_against,
            "n_catalog": self.n_catalog,
            "n_corpus": self.n_corpus,
            "findings": [finding.to_dict() for finding in self.findings],
            "skipped_corpus": self.skipped_corpus,
        }


def load_permitted_corpus(directory: Path | None = None) -> tuple[list[CorpusDocument], list[str]]:
    """Load only manifest-declared, permission-cleared documents.

    ``manifest.json`` format::

        {"documents": [{"file": "cc0_examples.md", "license": "CC0-1.0",
                        "permission_basis": "permissive_license",
                        "source": "https://example.org/dataset"}]}
    """
    directory = Path(directory or get_settings().permitted_corpora_dir)
    docs: list[CorpusDocument] = []
    skipped: list[str] = []
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return docs, skipped
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return docs, [f"manifest unreadable: {exc}"]
    for entry in manifest.get("documents", []):
        basis = str(entry.get("permission_basis", ""))
        license_text = str(entry.get("license", ""))
        file_name = str(entry.get("file", ""))
        if basis not in PERMITTED_BASES:
            skipped.append(f"{file_name}: permission_basis '{basis}' not permitted")
            continue
        if not license_text:
            skipped.append(f"{file_name}: missing license declaration")
            continue
        path = (directory / file_name).resolve()
        if directory.resolve() not in path.parents:
            skipped.append(f"{file_name}: path escapes the corpus directory")
            continue
        if not path.exists():
            skipped.append(f"{file_name}: file not found")
            continue
        docs.append(
            CorpusDocument(
                ref=f"{Path(file_name).name} ({license_text})",
                text=path.read_text(encoding="utf-8", errors="replace"),
                license=license_text,
                permission_basis=basis,
            )
        )
    return docs, skipped


def compare_against_catalog(session: Session, text: str) -> list[CorpusDocument]:
    rows = session.scalars(select(PromptVariant))
    return [
        CorpusDocument(
            ref=f"catalog:{row.id}:{row.model_family}",
            text=row.compiled_text,
            license="internal",
            permission_basis="internal",
        )
        for row in rows
    ]


def check_similarity(
    session: Session,
    *,
    text: str,
    exclude_ref: str | None = None,
    exclude_refs: set[str] | None = None,
    corpus: list[CorpusDocument] | None = None,
    include_catalog: bool = True,
) -> SimilarityResult:
    """Compare ``text`` against the catalog and permitted corpus.

    ``exclude_refs`` lets the gate ignore the version's own variants (otherwise every
    variant would trivially match its siblings).
    """
    settings = get_settings()
    embedder = get_embedder()
    candidates: list[CorpusDocument] = []
    if include_catalog:
        candidates.extend(compare_against_catalog(session, text))
    if corpus is None:
        corpus, skipped = load_permitted_corpus()
    else:
        skipped = []
    candidates.extend(corpus)
    blocked_refs = set(exclude_refs or set())
    if exclude_ref:
        blocked_refs.add(exclude_ref)
    candidates = [doc for doc in candidates if doc.ref not in blocked_refs and doc.text.strip()]

    if not candidates:
        return SimilarityResult(
            decision=str(GovernanceDecision.PASS),
            max_similarity=0.0,
            shingle_overlap=0.0,
            nearest_ref="",
            threshold=settings.ip_block_threshold,
            compared_against="none",
            skipped_corpus=skipped,
        )

    target_vector = embedder.embed([text])[0]
    target_shingles = shingles(text, size=5)
    max_similarity = 0.0
    max_overlap = 0.0
    nearest_ref = ""
    # Batch embeddings for the catalog/corpus in one call.
    vectors = embedder.embed([doc.text for doc in candidates])
    for doc, vector in zip(candidates, vectors, strict=True):
        similarity = cosine(target_vector, vector)
        overlap = jaccard(target_shingles, shingles(doc.text, size=5))
        if similarity > max_similarity:
            max_similarity, nearest_ref = similarity, doc.ref
        max_overlap = max(max_overlap, overlap)

    findings: list[Finding] = []
    decision = str(GovernanceDecision.PASS)
    if (
        max_similarity >= settings.ip_block_threshold
        or max_overlap >= settings.ip_shingle_overlap_threshold
    ):
        decision = str(GovernanceDecision.BLOCK)
        findings.append(
            Finding(
                category="ip:near_duplicate",
                severity="block",
                message=(
                    f"compiled prompt is too similar to {nearest_ref} "
                    f"(cosine {max_similarity:.2f}, shingle overlap {max_overlap:.2f})"
                ),
            )
        )
    elif max_similarity >= settings.ip_warn_threshold:
        decision = str(GovernanceDecision.NEEDS_REVIEW)
        findings.append(
            Finding(
                category="ip:similarity_warning",
                severity="warn",
                message=f"similar to {nearest_ref} (cosine {max_similarity:.2f}) — justify originality",
            )
        )

    return SimilarityResult(
        decision=decision,
        max_similarity=max_similarity,
        shingle_overlap=max_overlap,
        nearest_ref=nearest_ref,
        threshold=settings.ip_block_threshold,
        compared_against=f"catalog+corpus ({len(candidates)} docs)",
        n_catalog=len(compare_against_catalog(session, text)) if include_catalog else 0,
        n_corpus=len(corpus),
        findings=findings,
        skipped_corpus=skipped,
    )
