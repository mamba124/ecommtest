"""
Scraper utility — fetches documentation pages and saves them as Markdown.

URL discovery strategy (in order):
  1. Request /sitemap.xml → parse every <loc> that starts with ALLOWED_PREFIX.
  2. If sitemap is absent or yields no matching URLs, fall back to BFS link-following
     starting from the root docs page.

Page rendering strategy:
  1. Playwright (headless Chromium) — handles JavaScript-rendered (SPA/CSR) pages.
     Waits for networkidle + article/main selector before extracting HTML.
  2. httpx fallback — used when Playwright is unavailable (server-rendered pages only).
"""
import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup
import html2text

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger("scraper")

REQUEST_DELAY = 0.5          # seconds between requests (polite crawling)
DEFAULT_ALLOWED_PREFIX = "/en/docs/"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
_MIN_CONTENT_CHARS = 200     # pages shorter than this are likely JS placeholders


class DocScraper:
    def __init__(
        self,
        base_url: str,
        output_dir: str,
        max_pages: int = 50,
        use_playwright: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._allowed_prefix = DEFAULT_ALLOWED_PREFIX
        self._output_dir = Path(output_dir)
        self._max_pages = max_pages
        self._visited: set[str] = set()
        self._converter = html2text.HTML2Text()
        self._converter.ignore_links = False
        self._converter.body_width = 0
        self._use_playwright = use_playwright and _PLAYWRIGHT_AVAILABLE

        if use_playwright and not _PLAYWRIGHT_AVAILABLE:
            logger.warning(
                "playwright package not installed — falling back to httpx. "
                "JS-rendered pages (SPA/CSR) will likely return placeholder text. "
                "Install with: pip install playwright && playwright install chromium"
            )

    # ── Redirect resolution ───────────────────────────────────────────────────

    def _resolve_redirect(self, client: httpx.Client) -> None:
        """
        Probe the configured base_url + allowed_prefix and follow any permanent
        redirects (e.g. docs.anthropic.com → platform.claude.com).  Updates
        self._base_url and self._allowed_prefix in-place so all subsequent
        netloc/path filters use the real destination domain and path structure.
        """
        probe = f"{self._base_url}{self._allowed_prefix}"
        try:
            resp = client.head(probe, timeout=10.0)
            final = str(resp.url)
            parsed = urlparse(final)
            orig_netloc = urlparse(self._base_url).netloc
            if parsed.netloc and parsed.netloc != orig_netloc:
                self._base_url = f"{parsed.scheme}://{parsed.netloc}"
                # If the redirect landed on a specific page (no trailing slash),
                # strip the last path segment to get the docs root directory.
                path = parsed.path
                if not path.endswith("/"):
                    path = path.rsplit("/", 1)[0] + "/"
                self._allowed_prefix = path
                logger.info(
                    f"Redirect resolved: {orig_netloc} → {parsed.netloc} "
                    f"(new prefix: {self._allowed_prefix})"
                )
        except Exception as exc:
            logger.warning(f"Could not probe redirect for {probe}: {exc}")

    # ── URL discovery ─────────────────────────────────────────────────────────

    def _fetch_sitemap(self, client: httpx.Client) -> list[str]:
        """
        GET /sitemap.xml and return all <loc> URLs that fall under ALLOWED_PREFIX.
        Returns an empty list if the sitemap is missing, malformed, or contains
        no matching entries — caller should then fall back to link-following.
        """
        sitemap_url = f"{self._base_url}/sitemap.xml"
        try:
            resp = client.get(sitemap_url, timeout=15.0)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(f"Sitemap not available at {sitemap_url}: {exc}")
            return []

        try:
            root = ElementTree.fromstring(resp.text)
        except ElementTree.ParseError as exc:
            logger.warning(f"Failed to parse sitemap XML: {exc}")
            return []

        base_netloc = urlparse(self._base_url).netloc
        urls: list[str] = []

        # Handle both <urlset> and <sitemapindex> (index of sitemaps)
        for tag in root.iter(f"{{{SITEMAP_NS}}}loc"):
            loc = tag.text.strip() if tag.text else ""
            parsed = urlparse(loc)
            if parsed.netloc == base_netloc and parsed.path.startswith(self._allowed_prefix):
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                urls.append(clean)

        logger.info(f"Sitemap yielded {len(urls)} matching URLs under {self._allowed_prefix}")
        return urls

    def _find_links(self, html: str, current_url: str) -> list[str]:
        """Extract internal doc links from a page (used as BFS fallback)."""
        soup = BeautifulSoup(html, "html.parser")
        base_netloc = urlparse(self._base_url).netloc
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            absolute = urljoin(current_url, tag["href"])
            parsed = urlparse(absolute)
            if parsed.netloc == base_netloc and parsed.path.startswith(self._allowed_prefix):
                links.append(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
        return links

    # ── Content extraction ────────────────────────────────────────────────────

    def _slug(self, url: str) -> str:
        path = urlparse(url).path.strip("/").replace("/", "_")
        return path or "index"

    def _extract_content(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find("article") or soup.find("main") or soup.body
        if content is None:
            return ""
        return self._converter.handle(str(content))

    # ── robots.txt ────────────────────────────────────────────────────────────

    def _robots_ok(self, client: httpx.Client) -> bool:
        try:
            resp = client.get(f"{self._base_url}/robots.txt", timeout=10.0)
            if f"Disallow: {self._allowed_prefix}" in resp.text:
                logger.warning(f"robots.txt disallows scraping {self._allowed_prefix}")
                return False
        except Exception:
            pass
        return True

    # ── Page fetching backends ────────────────────────────────────────────────

    def _fetch_html_httpx(self, url: str, client: httpx.Client) -> str:
        resp = client.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.text

    def _fetch_html_playwright(self, url: str, pw_page) -> str:
        """
        Navigate to *url* with an already-open Playwright page.
        Waits for networkidle to let JS finish rendering, then waits for
        an article or main element to appear (content sentinel).
        Returns the fully-rendered page HTML.
        """
        pw_page.goto(url, wait_until="networkidle", timeout=30_000)
        # Give the content sentinel up to 10 s to appear
        try:
            pw_page.wait_for_selector("article, main", timeout=10_000)
        except PlaywrightTimeout:
            logger.debug(f"Timed out waiting for article/main on {url}")
        return pw_page.content()

    # ── Shared fetch loop ─────────────────────────────────────────────────────

    def _run_fetch_loop(
        self,
        queue: list[str],
        sitemap_urls: list[str],
        *,
        pw_page=None,
        http_client: httpx.Client | None = None,
    ) -> int:
        """
        Main crawl loop shared by both rendering backends.
        Pass either *pw_page* (Playwright) or *http_client* (httpx).
        """
        saved = 0
        while queue and len(self._visited) < self._max_pages:
            url = queue.pop(0)
            if url in self._visited:
                continue
            self._visited.add(url)

            try:
                if pw_page is not None:
                    html = self._fetch_html_playwright(url, pw_page)
                else:
                    html = self._fetch_html_httpx(url, http_client)

                content = self._extract_content(html)

                if len(content.strip()) < _MIN_CONTENT_CHARS:
                    logger.warning(
                        f"Skipping {url} — extracted content is only "
                        f"{len(content.strip())} chars (likely a JS placeholder). "
                        "Consider running with Playwright enabled."
                    )
                    continue

                out = self._output_dir / f"{self._slug(url)}.md"
                out.write_text(content, encoding="utf-8")
                saved += 1
                logger.info(f"Saved {url} → {out}")

                # In BFS fallback mode, discover links from each fetched page
                if not sitemap_urls:
                    for link in self._find_links(html, url):
                        if link not in self._visited:
                            queue.append(link)

                time.sleep(REQUEST_DELAY)

            except Exception as exc:
                logger.error(f"Failed to scrape {url}: {exc}")

        return saved

    # ── Main entry point ──────────────────────────────────────────────────────

    def scrape(self) -> int:
        """
        Scrape docs and return the number of pages saved.

        Phase 1 — URL discovery (always httpx, lightweight):
          1. /sitemap.xml  — parse <loc> entries matching ALLOWED_PREFIX.
          2. BFS fallback  — follow <a href> links starting from the root page.

        Phase 2 — Content fetching:
          1. Playwright (headless Chromium) if available and use_playwright=True.
          2. httpx fallback otherwise.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: URL discovery ────────────────────────────────────────────
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            self._resolve_redirect(client)

            if not self._robots_ok(client):
                logger.error("Scraping aborted per robots.txt")
                return 0

            sitemap_urls = self._fetch_sitemap(client)

            if sitemap_urls:
                queue = sitemap_urls[:self._max_pages]
                logger.info(
                    f"Using sitemap strategy: {len(queue)} URLs queued "
                    f"(max_pages={self._max_pages})"
                )
            else:
                queue = [f"{self._base_url}{self._allowed_prefix}"]
                logger.info("Sitemap unavailable — falling back to BFS link-following")

        # ── Phase 2: Content fetching ─────────────────────────────────────────
        if self._use_playwright:
            logger.info("Rendering with Playwright (headless Chromium)")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                try:
                    saved = self._run_fetch_loop(queue, sitemap_urls, pw_page=page)
                finally:
                    browser.close()
        else:
            logger.info("Fetching with httpx (Playwright unavailable or disabled)")
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                saved = self._run_fetch_loop(queue, sitemap_urls, http_client=client)

        logger.info(f"Scraping complete: {saved} pages saved to {self._output_dir}")
        return saved
