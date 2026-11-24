import json
from pathlib import Path

import httpx
import pytest

from ragmed_retrieval.corpus import make_id
from ragmed_retrieval.sources import (
    check_urls,
    current_address,
    load_source_urls,
    read_source_urls,
    safe_url,
    save_source_urls,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="1" source="GARD" url="{url}">
<QAPairs>
  <QAPair pid="1">
    <Question qid="1-1" qtype="information">What is  asthma? ?</Question>
    <Answer>Asthma   inflames the airways.</Answer>
  </QAPair>
  <QAPair pid="2"><Question qid="1-2" qtype="causes">Empty?</Question><Answer></Answer></QAPair>
</QAPairs>
</Document>"""


def write(folder: Path, name: str, url: str) -> None:
    (folder / "1_Test").mkdir(parents=True, exist_ok=True)
    (folder / "1_Test" / name).write_text(XML.format(url=url), encoding="utf-8")


def test_answers_are_matched_to_their_page_by_the_corpus_id(tmp_path: Path) -> None:
    write(tmp_path, "a.xml", "https://medlineplus.gov/asthma.html")
    write(tmp_path, "b.xml", "javascript:alert(1)")  # never a link
    asthma = make_id("What is asthma?", "Asthma inflames the airways.")

    assert read_source_urls(tmp_path) == {asthma: "https://medlineplus.gov/asthma.html"}
    assert read_source_urls(tmp_path, wanted={"other"}) == {}


def test_the_saved_map_loads_back_and_unsafe_links_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    save_source_urls({"a": "https://www.cancer.gov/x"}, path)
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "b": "file:///etc/passwd"}))

    assert load_source_urls(path) == {"a": "https://www.cancer.gov/x"}
    assert load_source_urls(tmp_path / "missing.json") == {}
    assert safe_url("ftp://x") == ""


def test_known_moves_are_applied() -> None:
    assert current_address("http://rarediseases.info.nih.gov/gard/10486/x") == (
        "https://rarediseases.info.nih.gov/diseases/10486/x"
    )
    assert current_address("http://www.nhlbi.nih.gov/a") == "https://www.nhlbi.nih.gov/a"


@pytest.mark.anyio
async def test_only_pages_that_still_load_are_kept_at_their_current_address() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        match request.url.path:
            case "/old":
                return httpx.Response(301, headers={"location": "https://site.gov/new"})
            case "/new":
                return httpx.Response(200)
            case "/home":
                return httpx.Response(301, headers={"location": "https://site.gov/"})
            case "/":
                return httpx.Response(200)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        alive = await check_urls(
            ["http://site.gov/old", "https://site.gov/gone", "https://site.gov/home",
             "http://www.cdc.gov/x"],
            client,
        )  # fmt: skip

    assert alive == {"http://site.gov/old": "https://site.gov/new"}
