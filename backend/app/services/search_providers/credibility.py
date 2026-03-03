"""Domain credibility scoring for source evaluation."""

from urllib.parse import urlparse

_TLD_SCORES: dict[str, float] = {
    ".gov": 0.9,
    ".edu": 0.9,
    ".org": 0.65,
}

_DOMAIN_SCORES: list[tuple[list[str], float]] = [
    (["nytimes.com", "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
      "washingtonpost.com", "theguardian.com", "bloomberg.com", "ft.com",
      "economist.com", "wsj.com"], 0.85),
    (["wikipedia.org"], 0.7),
    (["docs.python.org", "docs.microsoft.com", "developer.mozilla.org",
      "developer.apple.com", "cloud.google.com", "aws.amazon.com"], 0.8),
    (["stackoverflow.com", "stackexchange.com", "github.com", "gitlab.com"], 0.7),
    (["arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
      "semanticscholar.org", "nature.com", "science.org"], 0.85),
]


def score_domain(url: str) -> float:
    """Score a URL's domain credibility (0.0-1.0)."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return 0.5

    for tld, score in _TLD_SCORES.items():
        if hostname.endswith(tld):
            return score

    for domains, score in _DOMAIN_SCORES:
        for domain in domains:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return score

    if hostname.startswith("docs.") or hostname.startswith("developer."):
        return 0.8

    return 0.5
