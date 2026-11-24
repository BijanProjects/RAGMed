"""Where each MedQuAD answer was published: its page on an NIH website.

The Kaggle CSV the knowledge base is built from keeps only questions and answers. The original
release (https://github.com/abachaa/MedQuAD) is XML, one file per page, with the page's address.
Answer ids are a hash of the normalised question and answer (see corpus.make_id), so the two
releases can be matched exactly, without re-embedding anything. The result is a small JSON file,
answer id to URL, that the Medical Knowledge MCP server reads at startup.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import anyio
import httpx

from ragmed_retrieval.corpus import make_id, normalize_text

SOURCES_FILE = Path("data/raw/medquad-sources.json")
# Sites that no longer exist, or refuse every automated check, so a link cannot be verified.
UNCHECKABLE = {"nihseniorhealth.gov", "www.cdc.gov"}
CHECKS_AT_ONCE = 6  # be polite to the NIH sites
OK = 200


def current_address(url: str) -> str:
    """Known moves since the 2017 release: https everywhere; GARD pages moved to /diseases/."""
    url = url.replace("http://", "https://", 1)
    if urlparse(url).netloc == "rarediseases.info.nih.gov":
        url = url.replace("/gard/", "/diseases/", 1)
    return url


async def check_urls(urls: Iterable[str], client: httpx.AsyncClient) -> dict[str, str]:
    """Each URL that still loads, mapped to the address it ends up at after redirects.

    A page that only redirects to its site's home page counts as gone.
    """
    alive: dict[str, str] = {}
    limit = anyio.Semaphore(CHECKS_AT_ONCE)

    async def check(url: str) -> None:
        async with limit:
            try:
                response = await client.get(current_address(url), follow_redirects=True)
            except httpx.HTTPError:
                return
        final = str(response.url)
        if response.status_code == OK and urlparse(final).path.strip("/"):
            alive[url] = final

    async with anyio.create_task_group() as group:
        for url in sorted(set(urls)):
            if urlparse(url).netloc not in UNCHECKABLE:
                group.start_soon(check, url)
    return alive


def safe_url(url: str) -> str:
    """The URL if it is a plain web address (http or https with a host), else ""."""
    parsed = urlparse(url.strip())
    return url.strip() if parsed.scheme in ("http", "https") and parsed.netloc else ""


def read_source_urls(xml_dir: Path, wanted: Iterable[str] | None = None) -> dict[str, str]:
    """Answer id to page URL for every question-answer pair in the XML release.

    `wanted` keeps only those ids (the answers in the knowledge base).
    """
    keep = set(wanted) if wanted is not None else None
    urls: dict[str, str] = {}
    for path in sorted(xml_dir.glob("*/*.xml")):
        # The release is trusted research data, and ElementTree never fetches external entities.
        root = ElementTree.parse(path).getroot()  # noqa: S314
        url = safe_url(root.get("url", ""))
        if not url:
            continue
        for pair in root.iter("QAPair"):
            question = normalize_text(pair.findtext("Question") or "")
            answer = normalize_text(pair.findtext("Answer") or "")
            if not question or not answer:
                continue
            answer_id = make_id(question, answer)
            if keep is None or answer_id in keep:
                urls[answer_id] = url
    return urls


def save_source_urls(urls: dict[str, str], path: Path) -> None:
    path.write_text(json.dumps(dict(sorted(urls.items())), indent=0) + "\n", encoding="utf-8")


def load_source_urls(path: Path) -> dict[str, str]:
    """The saved map, or {} if there is none (sources then have no link)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): safe_url(str(v)) for k, v in data.items() if safe_url(str(v))}
