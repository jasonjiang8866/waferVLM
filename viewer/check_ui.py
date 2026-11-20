"""Exercise the deployed project path in Chrome (CI dependency: selenium)."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait

site = Path(sys.argv[1]).resolve()
server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=site.parent))
Thread(target=server.serve_forever, daemon=True).start()
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
with webdriver.Chrome(options=options) as browser:
    def find(selector):
        return browser.find_element(By.CSS_SELECTOR, selector)

    browser.get(f"http://127.0.0.1:{server.server_port}/{site.name}/")
    assert len(browser.find_elements(By.CSS_SELECTOR, "#thumbs button")) == 13
    Select(find("#classes")).select_by_value("3")
    assert len(browser.find_elements(By.CSS_SELECTOR, "#thumbs button")) == 2
    find("#thumbs button:last-child").click()
    for mode in ("mask", "overlay"):
        find(f"[data-mode='{mode}']").click()
        WebDriverWait(browser, 5).until(lambda b: b.execute_script(
            "return document.querySelector('#photo').naturalWidth === 480"))
    assert find("#tint").is_displayed() and find("#opacity").is_displayed()
    Select(find("#dataset")).select_by_value("silicon-wafer-pl")
    assert not find("[data-mode=mask]").is_enabled() and not find("#tint").is_displayed()
    browser.set_window_size(390, 844)
    assert browser.execute_script("return document.documentElement.scrollWidth <= innerWidth")
server.shutdown()
