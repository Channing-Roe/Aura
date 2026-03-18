# =============================================================================
# FILE: tools/browser.py
# AURA Browser Control via Playwright
#
# Replaces fragile screenshot → click loops with real browser automation.
# Works headless or headed. Supports login flows, form filling, scraping.
#
# Why Playwright over Selenium:
#   - Auto-waits for elements (no sleep hacks)
#   - Handles SPAs and modern JS
#   - Built-in network interception
#   - Async-native, faster
#
# Requirements:
#   pip install playwright
#   playwright install chromium
# =============================================================================

import os
import re
import logging
import asyncio
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Optional playwright import ────────────────────────────────────────────────
try:
    from playwright.async_api import (
        async_playwright, Browser, BrowserContext, Page,
        TimeoutError as PWTimeout
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright not installed. Run:\n"
        "  pip install playwright\n"
        "  playwright install chromium"
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BrowserResult:
    success:     bool
    output:      str
    url:         str        = ""
    title:       str        = ""
    screenshot:  bytes      = b""
    links:       List[str]  = field(default_factory=list)
    data:        Any        = None


# =============================================================================
# BROWSER SESSION
# Manages a long-lived browser instance for AURA
# =============================================================================

class BrowserSession:
    """
    A managed Playwright browser session.
    One session per AURA instance — pages are created and closed per task.

    Usage (sync wrapper — call from sync AURA code):
        session = BrowserSession()
        result  = session.run("go to google.com and search for AI news")
        print(result.output)
        session.close()
    """

    def __init__(self, headless: bool = True, slow_mo: int = 0):
        self._headless  = headless
        self._slow_mo   = slow_mo
        self._pw        = None
        self._browser: Optional[Browser]        = None
        self._context: Optional[BrowserContext] = None
        self._page:    Optional[Page]           = None
        self._lock      = threading.Lock()
        self._loop      = None
        self._thread    = None
        self._started   = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Launch the browser (call once before any run() calls)."""
        if self._started:
            return
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="BrowserLoop"
        )
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._async_start(), self._loop)
        future.result(timeout=30)
        self._started = True
        logger.info(f"Browser started (headless={self._headless})")

    def close(self):
        """Cleanly shut down the browser."""
        if not self._started:
            return
        future = asyncio.run_coroutine_threadsafe(self._async_close(), self._loop)
        try:
            future.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._started = False

    async def _async_start(self):
        self._pw      = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        self._page = await self._context.new_page()

    async def _async_close(self):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    # ── Sync wrappers ─────────────────────────────────────────────────────────

    def _run_async(self, coro, timeout: int = 30):
        """Run an async coroutine from sync code."""
        if not self._started:
            self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── Public API ────────────────────────────────────────────────────────────

    def goto(self, url: str, wait_for: str = "domcontentloaded") -> BrowserResult:
        """Navigate to a URL."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return self._run_async(self._async_goto(url, wait_for), timeout=30)

    def get_text(self) -> str:
        """Get the visible text content of the current page."""
        return self._run_async(self._async_get_text())

    def get_links(self) -> List[str]:
        """Get all links on the current page."""
        return self._run_async(self._async_get_links())

    def click(self, selector: str) -> BrowserResult:
        """Click an element by CSS selector or text."""
        return self._run_async(self._async_click(selector))

    def fill(self, selector: str, value: str) -> BrowserResult:
        """Fill an input field."""
        return self._run_async(self._async_fill(selector, value))

    def press(self, key: str):
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        return self._run_async(self._async_press(key))

    def screenshot(self) -> bytes:
        """Take a screenshot and return PNG bytes."""
        return self._run_async(self._async_screenshot())

    def scrape(self, url: str, css_selector: str = "body") -> BrowserResult:
        """Go to URL and extract text from a CSS selector."""
        return self._run_async(self._async_scrape(url, css_selector))

    def search_web(self, query: str, engine: str = "duckduckgo") -> BrowserResult:
        """Perform a web search and return the top results as text."""
        return self._run_async(self._async_search(query, engine))

    def run_task(self, prompt: str, context: str = "") -> BrowserResult:
        """
        High-level: interpret a natural language browser task and execute it.
        Examples:
          "go to github.com and search for python projects"
          "fill in the login form with user@example.com and submit"
        """
        return self._run_async(self._async_run_task(prompt), timeout=60)

    # ── Async implementations ─────────────────────────────────────────────────

    async def _async_goto(self, url: str, wait_for: str) -> BrowserResult:
        try:
            resp = await self._page.goto(url, wait_until=wait_for, timeout=20000)
            title = await self._page.title()
            return BrowserResult(
                success=True,
                output=f"Navigated to: {title}",
                url=self._page.url,
                title=title
            )
        except PWTimeout:
            return BrowserResult(success=False, output=f"Timeout navigating to {url}", url=url)
        except Exception as e:
            return BrowserResult(success=False, output=f"Navigation error: {e}", url=url)

    async def _async_get_text(self) -> str:
        try:
            return await self._page.inner_text("body")
        except Exception:
            return ""

    async def _async_get_links(self) -> List[str]:
        try:
            handles = await self._page.query_selector_all("a[href]")
            links = []
            for h in handles[:50]:
                href = await h.get_attribute("href")
                if href and href.startswith("http"):
                    links.append(href)
            return links
        except Exception:
            return []

    async def _async_click(self, selector: str) -> BrowserResult:
        try:
            # Try CSS selector first, then text
            try:
                await self._page.click(selector, timeout=5000)
            except Exception:
                await self._page.get_by_text(selector).first.click(timeout=5000)
            await self._page.wait_for_load_state("domcontentloaded", timeout=10000)
            return BrowserResult(success=True, output=f"Clicked: {selector}", url=self._page.url)
        except Exception as e:
            return BrowserResult(success=False, output=f"Click failed: {e}")

    async def _async_fill(self, selector: str, value: str) -> BrowserResult:
        try:
            await self._page.fill(selector, value, timeout=5000)
            return BrowserResult(success=True, output=f"Filled '{selector}' with value")
        except Exception as e:
            return BrowserResult(success=False, output=f"Fill failed: {e}")

    async def _async_press(self, key: str):
        await self._page.keyboard.press(key)

    async def _async_screenshot(self) -> bytes:
        try:
            return await self._page.screenshot(type="png")
        except Exception:
            return b""

    async def _async_scrape(self, url: str, selector: str) -> BrowserResult:
        nav = await self._async_goto(url, "domcontentloaded")
        if not nav.success:
            return nav
        try:
            # Wait for content
            await self._page.wait_for_selector(selector, timeout=8000)
            text = await self._page.inner_text(selector)
            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            title = await self._page.title()
            return BrowserResult(
                success=True,
                output=text[:8000],
                url=self._page.url,
                title=title
            )
        except Exception as e:
            # Fallback: whole page text
            text = await self._async_get_text()
            return BrowserResult(
                success=True,
                output=(text[:8000] if text else f"Scrape error: {e}"),
                url=self._page.url
            )

    async def _async_search(self, query: str, engine: str) -> BrowserResult:
        engines = {
            "duckduckgo": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
            "google":     f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "bing":       f"https://www.bing.com/search?q={query.replace(' ', '+')}",
        }
        url = engines.get(engine, engines["duckduckgo"])
        nav = await self._async_goto(url, "networkidle")
        if not nav.success:
            return nav
        text = await self._async_get_text()
        # Extract snippet-like content (first 4000 chars is usually enough)
        return BrowserResult(
            success=True,
            output=text.output[:4000] if isinstance(text, BrowserResult) else str(text)[:4000],
            url=self._page.url,
            title=await self._page.title()
        )

    async def _async_run_task(self, prompt: str) -> BrowserResult:
        """
        Parse a natural language browser command and execute the right action.
        This is a lightweight task runner — not a full LLM agent loop.
        """
        low = prompt.lower()

        # "go to / open / visit <url>"
        m = re.search(r'(?:go to|open|visit|navigate to)\s+([\w.\-/]+)', low)
        if m:
            return await self._async_goto("https://" + m.group(1), "domcontentloaded")

        # "search for <query> on <engine>"
        m = re.search(r'search (?:for )?(.+?)(?:\s+on\s+(google|duckduckgo|bing))?$', low)
        if m:
            engine = m.group(2) or "duckduckgo"
            return await self._async_search(m.group(1).strip(), engine)

        # "scrape / get / read <url>"
        m = re.search(r'(?:scrape|get|read|fetch)\s+(https?://\S+)', low)
        if m:
            return await self._async_scrape(m.group(1), "body")

        # "click <selector or text>"
        m = re.search(r'click\s+(.+)', low)
        if m:
            return await self._async_click(m.group(1).strip())

        # "type <value> into <selector>"
        m = re.search(r'type\s+(.+?)\s+into\s+(.+)', low)
        if m:
            return await self._async_fill(m.group(2).strip(), m.group(1).strip())

        # "screenshot / take a screenshot"
        if "screenshot" in low:
            png = await self._async_screenshot()
            return BrowserResult(
                success=bool(png),
                output="Screenshot taken" if png else "Screenshot failed",
                screenshot=png
            )

        return BrowserResult(
            success=False,
            output=f"Didn't understand browser task: '{prompt}'"
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_session: Optional[BrowserSession] = None


def get_browser(headless: bool = True) -> BrowserSession:
    """Get or create the shared BrowserSession."""
    global _session
    if _session is None:
        _session = BrowserSession(headless=headless)
    if not _session._started:
        _session.start()
    return _session


def browser_task(prompt: str, context: str = "") -> BrowserResult:
    """Convenience function for the decision system / tool executor."""
    if not PLAYWRIGHT_AVAILABLE:
        return BrowserResult(
            success=False,
            output=(
                "Playwright not installed.\n"
                "Fix: pip install playwright && playwright install chromium"
            )
        )
    return get_browser().run_task(prompt, context)


def browser_scrape(url: str) -> BrowserResult:
    """Scrape a URL and return its text content."""
    if not PLAYWRIGHT_AVAILABLE:
        return BrowserResult(success=False, output="Playwright not installed")
    return get_browser().scrape(url)


def browser_search(query: str) -> BrowserResult:
    """Search DuckDuckGo and return the results page text."""
    if not PLAYWRIGHT_AVAILABLE:
        return BrowserResult(success=False, output="Playwright not installed")
    return get_browser().search_web(query)
