from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.core.config import settings
from app.rag.models import RetrievedChunk, SourceType


logger = logging.getLogger(__name__)

# DuckDuckGo HTML search returns a bot interstitial (HTTP 202, 0 results)
# for non-browser User-Agents. Use a normal browser UA for the free fallback.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_OFFICIAL_HOST_SUFFIXES = (
    "pakistancode.gov.pk",
    "punjablaws.gov.pk",
    "sindhlaws.gov.pk",
    "na.gov.pk",
    "senate.gov.pk",
    "supremecourt.gov.pk",
    "lhc.gov.pk",
    "shc.gov.pk",
    "ihc.gov.pk",
    "phc.gov.pk",
    "bhc.gov.pk",
    "federalshariatcourt.gov.pk",
    "molaw.gov.pk",
    "pakistan.gov.pk",
)

_RESULT_LINK = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_RESULT_SNIPPET = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>'
    r'|<(?:td|div)[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</(?:td|div)>',
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str
    score: float
    official: bool = False


class WebSearchService:
    """
    Live web search for legal research.

    Providers:
      tavily  — preferred for RAG (requires WEB_SEARCH_API_KEY / TAVILY_API_KEY)
      brave   — Brave Search API (BRAVE_API_KEY or WEB_SEARCH_API_KEY)
      duckduckgo — HTML fallback, no API key
      auto    — first configured paid provider, else DuckDuckGo
    """

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> list[WebSearchHit]:
        if not settings.WEB_SEARCH_ENABLED:
            return []
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        limit = max(1, min(limit or settings.WEB_SEARCH_MAX_RESULTS, 8))
        provider = self.resolved_provider()
        legal_query = self._legal_query(cleaned)

        try:
            if provider == "tavily":
                hits = await self._search_tavily(legal_query, limit=limit)
            elif provider == "brave":
                hits = await self._search_brave(legal_query, limit=limit)
            else:
                hits = await self._search_duckduckgo(legal_query, limit=limit)
        except Exception:
            logger.exception("Web search failed provider=%s", provider)
            if provider != "duckduckgo":
                try:
                    hits = await self._search_duckduckgo(legal_query, limit=limit)
                except Exception:
                    logger.exception("DuckDuckGo fallback search failed")
                    return []
            else:
                return []

        return hits[:limit]

    async def search_as_chunks(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        hits = await self.search(query, limit=limit)
        return [self.to_chunk(hit, index) for index, hit in enumerate(hits, 1)]

    def resolved_provider(self) -> str:
        requested = (settings.WEB_SEARCH_PROVIDER or "auto").strip().lower()
        key = (settings.resolved_web_search_api_key or "").strip()
        if requested in {"tavily", "brave", "duckduckgo"}:
            if requested in {"tavily", "brave"} and not key:
                return "duckduckgo"
            return requested
        if key and (settings.TAVILY_API_KEY or "").strip():
            return "tavily"
        if key and (settings.BRAVE_API_KEY or "").strip():
            return "brave"
        if key:
            return "tavily"
        return "duckduckgo"

    def to_chunk(self, hit: WebSearchHit, index: int) -> RetrievedChunk:
        host = urlparse(hit.url).netloc
        text = hit.snippet.strip() or hit.title
        title = hit.title.strip() or host or "Web result"
        return RetrievedChunk(
            id=f"web:{index}:{hit.url}",
            score=hit.score,
            relevance_score=hit.score,
            text=text,
            filename=title,
            title=title,
            display_name=title,
            document_id=hit.url,
            source="web",
            source_type=SourceType.WEB.value,
            source_reference=hit.url,
            url=hit.url,
            document_type="official_web" if hit.official else "web",
            jurisdiction="pakistan",
            summary=hit.snippet[:400] if hit.snippet else None,
        )

    @staticmethod
    def is_official_url(url: str) -> bool:
        host = (urlparse(url).netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in _OFFICIAL_HOST_SUFFIXES
        )

    @staticmethod
    def _legal_query(query: str) -> str:
        lower = query.lower()
        if "pakistan" in lower or "pakistani" in lower:
            return query
        return f"{query} Pakistan law"

    async def _search_tavily(self, query: str, *, limit: int) -> list[WebSearchHit]:
        api_key = settings.resolved_web_search_api_key
        if not api_key:
            return await self._search_duckduckgo(query, limit=limit)
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": limit,
        }
        async with httpx.AsyncClient(timeout=settings.WEB_SEARCH_TIMEOUT) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        hits: list[WebSearchHit] = []
        for index, item in enumerate(data.get("results") or []):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            title = str(item.get("title") or url)
            snippet = str(item.get("content") or item.get("snippet") or "")
            score = float(item.get("score") or max(0.2, 0.92 - index * 0.08))
            hits.append(
                WebSearchHit(
                    title=title,
                    url=url,
                    snippet=snippet[:2000],
                    score=min(1.0, score),
                    official=self.is_official_url(url),
                )
            )
        return hits

    async def _search_brave(self, query: str, *, limit: int) -> list[WebSearchHit]:
        api_key = settings.resolved_web_search_api_key or settings.BRAVE_API_KEY
        if not api_key:
            return await self._search_duckduckgo(query, limit=limit)
        async with httpx.AsyncClient(timeout=settings.WEB_SEARCH_TIMEOUT) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": limit},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
            )
            response.raise_for_status()
            data = response.json()
        hits: list[WebSearchHit] = []
        results = (data.get("web") or {}).get("results") or []
        for index, item in enumerate(results):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            title = str(item.get("title") or url)
            snippet = str(item.get("description") or "")
            hits.append(
                WebSearchHit(
                    title=title,
                    url=url,
                    snippet=snippet[:2000],
                    score=max(0.2, 0.9 - index * 0.08),
                    official=self.is_official_url(url),
                )
            )
        return hits

    async def _search_duckduckgo(self, query: str, *, limit: int) -> list[WebSearchHit]:
        # trust_env=False avoids corporate HTTP(S)_PROXY 403s that look like
        # "0 hits" when the free HTML fallback is the only provider.
        async with httpx.AsyncClient(
            timeout=settings.WEB_SEARCH_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            trust_env=False,
        ) as client:
            response = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
            )
            response.raise_for_status()
            html = response.text

        titles_and_urls = _RESULT_LINK.findall(html)
        if not titles_and_urls:
            lower = html.lower()
            if response.status_code == 202 or "anomaly" in lower or "bot" in lower:
                logger.warning(
                    "DuckDuckGo returned no parseable results (status=%s); "
                    "set TAVILY_API_KEY or BRAVE_API_KEY for reliable web search",
                    response.status_code,
                )
            return []

        snippets = [
            unescape(_TAG.sub(" ", match[0] or match[1] or ""))
            for match in _RESULT_SNIPPET.findall(html)
        ]
        hits: list[WebSearchHit] = []
        seen: set[str] = set()
        for index, (raw_href, raw_title) in enumerate(titles_and_urls):
            url = self._unwrap_duckduckgo_url(unescape(raw_href))
            if not url or url in seen or url.startswith("https://duckduckgo.com"):
                continue
            seen.add(url)
            title = _WHITESPACE.sub(
                " ",
                unescape(_TAG.sub(" ", raw_title)),
            ).strip()
            snippet = ""
            if index < len(snippets):
                snippet = _WHITESPACE.sub(" ", snippets[index]).strip()
            hits.append(
                WebSearchHit(
                    title=title or url,
                    url=url,
                    snippet=snippet[:2000],
                    score=max(0.2, 0.85 - index * 0.07),
                    official=self.is_official_url(url),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    @staticmethod
    def _unwrap_duckduckgo_url(href: str) -> str:
        if not href:
            return ""
        parsed = urlparse(href)
        if "uddg" in (parsed.query or ""):
            values = parse_qs(parsed.query).get("uddg") or []
            if values:
                return unquote(values[0])
        if href.startswith("//"):
            href = "https:" + href
            parsed = urlparse(href)
            if "uddg" in (parsed.query or ""):
                values = parse_qs(parsed.query).get("uddg") or []
                if values:
                    return unquote(values[0])
        return href
