from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


USER = {
    "id": "7c03f4d9-4434-4632-9e74-52515db323bf",
    "org_id": "acme-org",
    "department_id": "operations",
    "email": "admin@example.com",
    "display_name": "系统管理员",
    "role": "admin",
}

KNOWLEDGE_BASES = {
    "items": [
        {
            "id": "kb-operations",
            "name": "运营知识库",
            "description": "运行手册与常见问题",
            "file_count": 18,
            "completed_file_count": 17,
            "failed_job_count": 1,
            "can_read": True,
            "can_write": True,
            "has_full_access": True,
        }
    ]
}


def route_api(page: Page) -> None:
    page.route("**/api/auth/me", lambda route: route.fulfill(json=USER))
    page.route("**/api/knowledge-bases", lambda route: route.fulfill(json=KNOWLEDGE_BASES))
    page.route("**/api/health", lambda route: route.fulfill(json={"status": "ok", "ready": True}))
    page.route("**/api/api-keys", lambda route: route.fulfill(json={"items": []}))


def assert_overview(page: Page) -> None:
    page.wait_for_load_state("networkidle")
    page.get_by_role("heading", name="运营总览").wait_for()
    page.get_by_text("服务就绪", exact=True).wait_for()
    page.locator(".table-primary strong", has_text="运营知识库").wait_for()
    workspace_width = page.locator(".workspace").evaluate("node => node.getBoundingClientRect().width")
    if workspace_width < 300:
        raise AssertionError(f"Console workspace is too narrow: {workspace_width}px")
    overflow = page.locator("body").evaluate(
        """node => ({
          scrollWidth: node.scrollWidth,
          viewportWidth: window.innerWidth,
          layout: ['.app-shell', '.app-sider', '.workspace', '.overview-page'].map(selector => {
            const item = document.querySelector(selector);
            const rect = item?.getBoundingClientRect();
            return [selector, rect?.left, rect?.right, rect?.width];
          }),
          elements: [...document.querySelectorAll('*')]
            .filter(item => item.getBoundingClientRect().right > window.innerWidth + 1)
            .slice(0, 8)
            .map(item => `${item.tagName}.${item.className}`),
        })"""
    )
    if overflow["scrollWidth"] > overflow["viewportWidth"]:
        raise AssertionError(f"Console has horizontal overflow: {overflow}")


def main() -> None:
    output_dir = Path("/tmp")
    base_url = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:5173")
    stored_user = json.dumps(USER)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(viewport={"width": 1440, "height": 900})
        desktop.add_init_script(
            """
            localStorage.setItem('rag_token', 'smoke-token');
            localStorage.setItem('rag_user', JSON.stringify(%s));
            """ % stored_user
        )
        route_api(desktop)
        desktop.goto(base_url)
        assert_overview(desktop)
        desktop.screenshot(path=str(output_dir / "rag-console-desktop.png"), full_page=True)
        desktop.get_by_role("menuitem", name="开放接口").click()
        desktop.get_by_role("heading", name="开放接口").wait_for()

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.add_init_script(
            """
            localStorage.setItem('rag_token', 'smoke-token');
            localStorage.setItem('rag_user', JSON.stringify(%s));
            """ % stored_user
        )
        route_api(mobile)
        mobile.goto(base_url)
        assert_overview(mobile)
        mobile.screenshot(path=str(output_dir / "rag-console-mobile.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
