"""
Helpers for the AI competitive readout POC page.

This module keeps scraping and Gemini orchestration separate from the
core Monte Carlo simulator so the existing app flow remains unchanged.
"""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse
from dataclasses import asdict, dataclass
from typing import Any

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

CTG_API_BASE = "https://clinicaltrials.gov/api/v2"
CTG_STUDY_FIELDS = [
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.descriptionModule",
    "protocolSection.conditionsModule",
    "protocolSection.designModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule",
    "protocolSection.referencesModule",
    "derivedSection.miscInfoModule",
    "hasResults",
]


@dataclass
class EvidenceSource:
    """Normalized evidence item passed to the model and UI."""

    source_id: str
    source_type: str
    title: str
    url: str | None
    content: str
    snippets: list[str]
    status: str = "ok"
    error: str | None = None

    def to_model_block(self) -> str:
        """Render the source into a compact block for the model prompt."""
        lines = [
            f"[{self.source_id}]",
            f"Type: {self.source_type}",
            f"Title: {self.title or 'Untitled source'}",
        ]
        if self.url:
            lines.append(f"URL: {self.url}")
        if self.snippets:
            lines.append("Key excerpts:")
            lines.extend(f"- {snippet}" for snippet in self.snippets)
        lines.append("Full extracted content:")
        lines.append(self.content)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def extract_urls(raw_text: str) -> list[str]:
    """Extract unique URLs in their original order."""
    matches = re.findall(r"https?://[^\s,\]>\"')]+", raw_text or "")
    seen: set[str] = set()
    urls: list[str] = []
    for url in matches:
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_nct_ids(raw_text: str) -> list[str]:
    """Extract unique NCT IDs from free text."""
    matches = re.findall(r"\bNCT0*[1-9]\d{0,7}\b", raw_text or "", flags=re.IGNORECASE)
    seen: set[str] = set()
    nct_ids: list[str] = []
    for match in matches:
        normalized = match.upper()
        if normalized not in seen:
            seen.add(normalized)
            nct_ids.append(normalized)
    return nct_ids


def _extract_nct_id_from_url(url: str) -> str | None:
    match = re.search(r"\b(NCT0*[1-9]\d{0,7})\b", url, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _is_clinicaltrials_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return "clinicaltrials.gov" in host


def _pick_snippets(paragraphs: list[str], max_snippets: int = 3) -> list[str]:
    filtered = [_clean_text(p) for p in paragraphs if len(_clean_text(p)) >= 80]
    return filtered[:max_snippets]


def scrape_url(url: str, source_id: str, timeout: int = 12, max_chars: int = 12000) -> EvidenceSource:
    """Fetch a page and extract a readable text payload."""
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        return EvidenceSource(
            source_id=source_id,
            source_type="web",
            title=url,
            url=url,
            content="",
            snippets=[],
            status="error",
            error=str(exc),
        )

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else url)
    paragraph_nodes = soup.find_all(["p", "li", "h1", "h2", "h3"])
    paragraphs = [_clean_text(node.get_text(" ", strip=True)) for node in paragraph_nodes]
    paragraphs = [p for p in paragraphs if p]

    if not paragraphs:
        body_text = _clean_text(soup.get_text(" ", strip=True))
        paragraphs = [body_text] if body_text else []

    snippets = _pick_snippets(paragraphs)
    content = "\n".join(paragraphs)
    content = content[:max_chars]

    if not content:
        return EvidenceSource(
            source_id=source_id,
            source_type="web",
            title=title or url,
            url=url,
            content="No readable content could be extracted from this page.",
            snippets=[],
            status="error",
            error="No readable content found on page.",
        )

    return EvidenceSource(
        source_id=source_id,
        source_type="web",
        title=title or url,
        url=url,
        content=content,
        snippets=snippets,
    )


def _join_list(values: list[str] | None) -> str:
    return ", ".join(v for v in (values or []) if v)


def fetch_clinicaltrials_study(nct_id: str, source_id: str, timeout: int = 12, max_chars: int = 12000) -> EvidenceSource:
    """Fetch a ClinicalTrials.gov study through the official v2 API."""
    normalized_id = nct_id.upper()
    url = f"{CTG_API_BASE}/studies/{normalized_id}"
    params = {
        "format": "json",
        "markupFormat": "markdown",
        "fields": ",".join(CTG_STUDY_FIELDS),
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        study = response.json()
    except (requests.RequestException, ValueError) as exc:
        return EvidenceSource(
            source_id=source_id,
            source_type="clinicaltrials_gov",
            title=f"ClinicalTrials.gov study {normalized_id}",
            url=f"https://clinicaltrials.gov/study/{normalized_id}",
            content="",
            snippets=[],
            status="error",
            error=str(exc),
        )

    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    description = protocol.get("descriptionModule", {})
    conditions = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    references = protocol.get("referencesModule", {})

    lines = [
        f"NCT ID: {ident.get('nctId', normalized_id)}",
        f"Brief title: {ident.get('briefTitle', 'Unknown')}",
        f"Official title: {ident.get('officialTitle', 'Unknown')}",
        f"Overall status: {status.get('overallStatus', 'Unknown')}",
        f"Study start date: {status.get('startDateStruct', {}).get('date', 'Unknown')}",
        f"Primary completion date: {status.get('primaryCompletionDateStruct', {}).get('date', 'Unknown')}",
        f"Completion date: {status.get('completionDateStruct', {}).get('date', 'Unknown')}",
        f"Last update posted: {status.get('lastUpdatePostDateStruct', {}).get('date', 'Unknown')}",
        f"Has results: {study.get('hasResults', False)}",
        f"Lead sponsor: {sponsor.get('leadSponsor', {}).get('name', 'Unknown')}",
        f"Phases: {_join_list(design.get('phases')) or 'Unknown'}",
        f"Study type: {design.get('studyType', 'Unknown')}",
        f"Enrollment: {design.get('enrollmentInfo', {}).get('count', 'Unknown')} ({design.get('enrollmentInfo', {}).get('type', 'Unknown')})",
        f"Conditions: {_join_list(conditions.get('conditions')) or 'Unknown'}",
        f"Keywords: {_join_list(conditions.get('keywords')) or 'None listed'}",
        f"Brief summary: {description.get('briefSummary', 'Not provided')}",
        f"Detailed description: {description.get('detailedDescription', 'Not provided')}",
    ]

    primary_outcomes = outcomes.get("primaryOutcomes", [])[:3]
    for idx, outcome in enumerate(primary_outcomes, start=1):
        lines.append(
            f"Primary outcome {idx}: {outcome.get('measure', 'Unknown')} | Time frame: {outcome.get('timeFrame', 'Unknown')}"
        )

    arm_groups = arms.get("armGroups", [])[:4]
    for idx, arm in enumerate(arm_groups, start=1):
        lines.append(
            f"Arm {idx}: {arm.get('label', 'Unknown')} | Type: {arm.get('type', 'Unknown')} | Interventions: {_join_list(arm.get('interventionNames')) or 'Unknown'}"
        )

    refs = references.get("references", [])[:3]
    for idx, ref in enumerate(refs, start=1):
        lines.append(f"Reference {idx}: {ref.get('citation', 'Unknown')}")

    content = "\n".join(lines)[:max_chars]
    snippets = [
        line for line in lines
        if any(
            key in line.lower()
            for key in ["overall status", "primary completion date", "completion date", "enrollment", "primary outcome", "has results"]
        )
    ][:5]

    title = ident.get("briefTitle") or f"ClinicalTrials.gov study {normalized_id}"
    return EvidenceSource(
        source_id=source_id,
        source_type="clinicaltrials_gov",
        title=title,
        url=f"https://clinicaltrials.gov/study/{normalized_id}",
        content=content,
        snippets=snippets,
    )


def notes_source(notes: str, source_id: str = "N1", max_chars: int = 8000) -> EvidenceSource:
    """Wrap pasted analyst notes into the same evidence structure."""
    cleaned = _clean_text(notes)
    snippets = [cleaned[:240]] if cleaned else []
    return EvidenceSource(
        source_id=source_id,
        source_type="analyst_note",
        title="Analyst-provided notes",
        url=None,
        content=cleaned[:max_chars],
        snippets=snippets,
    )


def gather_sources(urls: list[str], analyst_notes: str | None = None, nct_ids: list[str] | None = None) -> list[EvidenceSource]:
    """Build the source bundle that will be shown and passed to Gemini."""
    sources: list[EvidenceSource] = []
    seen_ctg_ids: set[str] = set()

    for nct_id in nct_ids or []:
        normalized = nct_id.upper()
        if normalized in seen_ctg_ids:
            continue
        seen_ctg_ids.add(normalized)
        sources.append(fetch_clinicaltrials_study(normalized, source_id=f"S{len(sources) + 1}"))

    for url in urls:
        ctg_id = _extract_nct_id_from_url(url) if _is_clinicaltrials_url(url) else None
        if ctg_id:
            if ctg_id in seen_ctg_ids:
                continue
            seen_ctg_ids.add(ctg_id)
            sources.append(fetch_clinicaltrials_study(ctg_id, source_id=f"S{len(sources) + 1}"))
        else:
            sources.append(scrape_url(url, source_id=f"S{len(sources) + 1}"))

    if analyst_notes and analyst_notes.strip():
        sources.append(notes_source(analyst_notes))

    return sources


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "confidence": {"type": "string"},
            "summary_with_citations": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "impact": {"type": "string"},
                        "citation_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["claim", "impact", "citation_ids"],
                },
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "decision",
            "confidence",
            "summary_with_citations",
            "claims",
            "limitations",
            "follow_up_questions",
        ],
    }


def analyze_sources(
    analysis_question: str,
    sources: list[EvidenceSource],
    api_key: str | None = None,
    model_name: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """
    Ask Gemini to make a directional decision using only supplied evidence.

    Returns a parsed JSON object suitable for Streamlit rendering.
    """
    usable_sources = [source for source in sources if source.content]
    if not usable_sources:
        raise ValueError("No usable evidence was provided to the model.")

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or enter it in the page sidebar.")

    evidence_bundle = "\n\n".join(source.to_model_block() for source in usable_sources)
    prompt = f"""
You are a biotech competitive-intelligence analyst helping forecast trial readouts.

Your task:
- Answer the user question using ONLY the evidence provided below.
- Make an informed, directional decision.
- Cite evidence with source IDs like [S1] or [N1].
- If evidence is weak, say so clearly.
- Do not invent facts that are not in the sources.

User question:
{analysis_question}

Required output rules:
- `summary_with_citations` must read naturally and include inline source IDs.
- Every claim must include at least one citation ID.
- `impact` should explain why that claim changes the readout outlook or decision.
- `confidence` must be one of: high, medium, low.

Evidence:
{evidence_bundle}
""".strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_json_schema=_response_schema(),
        ),
    )

    response_text = _strip_code_fences(response.text or "")
    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    parsed = json.loads(response_text)
    parsed["sources"] = [source.to_dict() for source in sources]
    return parsed
