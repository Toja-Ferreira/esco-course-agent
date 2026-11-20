"""Record the portfolio demo: one real session, captured at 1920x1080 and 60 fps.

Chrome runs full-screen (kiosk, 1.5x pixel density: the page's 1280x720 layout,
drawn sharp at 1080p) on a virtual display (Xvfb), and ffmpeg records that
display. The web app runs standalone (no login) on a free local port, in a
temporary folder, so the demo session never lands in results/. Only
DEEPSEEK_API_KEY, HF_TOKEN and DEEPSEEK_MODEL are read from .env; nothing from it
is printed.

Run from the repo root, with data/ in place:
    sudo apt install xvfb ffmpeg
    pip install playwright && python -m playwright install chromium
    python demo/record.py
Writes demo/out/raw.mkv and demo/out/timeline.json. Then: python demo/edit.py

The story: paste the course page -> skills with evidence -> keep them, remove one
with a reason -> occupations -> what the course lacks for one -> text for a
missing skill, checked -> accept it, and the re-check finds what else it shows.

The course text (demo/course.txt) is adapted from "Learn to code for data
analysis", The Open University (OpenLearn), CC BY-NC-SA:
https://www.open.edu/openlearn/digital-computing/learn-code-data-analysis/content-section-overview-0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo" / "out"
COURSE_NAME = "Learn to code for data analysis"
SOURCE = ("https://www.open.edu/openlearn/digital-computing/learn-code-data-analysis/"
          "content-section-overview-0")
KEEP = "Keep all the suggested ones."
REMOVE_NOTE = "too generic for this course"
ASK_OCC = "Which occupations could this course lead to?"
ASK_LACK = "What is missing for {occ}?"
ASK_DRAFT = 'Suggest text the course could add so that it teaches "{skill}".'
ACCEPT = "I accept that change."
PREFERRED_OCC = "data analyst"
TYPE_DELAY_MS = 40
REPLY_TIMEOUT_MS = 5 * 60 * 1000
DISPLAY = ":99"
SIZE = (1920, 1080)          # recorded pixels
SCALE = 1.5                  # the page is laid out at SIZE / SCALE = 1280x720


class TakeFailed(RuntimeError):
    """The session went somewhere the demo can't use; run it again."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_env(port: int) -> dict:
    from dotenv import dotenv_values
    keys = dotenv_values(REPO / ".env")
    env = dict(os.environ)
    for k in ("DEEPSEEK_API_KEY", "HF_TOKEN", "DEEPSEEK_MODEL"):
        if keys.get(k) and not env.get(k):
            env[k] = keys[k]
    if not env.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY is not set (in the environment or .env).")
    # standalone: no POF login, no university login; set (even empty) so .env can't override
    env.update(ESCO_AUTH="", UA_OAUTH_CLIENT_ID="", UA_OAUTH_CLIENT_SECRET="",
               ESCO_DATA_DIR=str(REPO / "data"), ESCO_SALT="demo",
               HOST="127.0.0.1", PORT=str(port))
    return env


def wait_up(url: str, proc: subprocess.Popen, log: Path, seconds: int = 180) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            sys.exit(f"The server stopped. Its log:\n{log.read_text()[-3000:]}")
        try:
            urllib.request.urlopen(url + "api/me", timeout=2)
            return
        except Exception:
            time.sleep(1)
    sys.exit("The server did not start in time.")


def browser(p, profile: Path):
    """Full-screen Chrome on the virtual display: no address bar, tabs or pop-ups."""
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Preferences").write_text(json.dumps({"translate": {"enabled": False}}))
    w, h = int(SIZE[0] / SCALE), int(SIZE[1] / SCALE)
    return p.chromium.launch_persistent_context(
        str(profile), headless=False, no_viewport=True, env={**os.environ, "DISPLAY": DISPLAY},
        ignore_default_args=["--enable-automation"],
        args=["--kiosk", f"--force-device-scale-factor={SCALE}", "--window-position=0,0",
              f"--window-size={w},{h}", "--hide-scrollbars", "--disable-infobars",
              "--disable-features=Translate,TranslateUI", "--lang=en-GB", "--no-first-run",
              "--no-default-browser-check", "--disable-notifications"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--occupation", default=PREFERRED_OCC,
                    help="occupation to check for gaps, if it is ranked (else the first one)")
    ap.add_argument("--skill", help="skill for the text suggestion (default: the first "
                                    "missing skill the gap check lists)")
    a = ap.parse_args()
    if not (REPO / "data").is_dir():
        sys.exit("data/ is missing: the ESCO parquet files and embeddings are needed.")
    for tool in ("Xvfb", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is missing: sudo apt install xvfb ffmpeg")

    from playwright.sync_api import sync_playwright

    course = (REPO / "demo" / "course.txt").read_text().strip()
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="esco-demo-"))
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    log = work / "server.log"
    procs: list[subprocess.Popen] = []
    server = subprocess.Popen([sys.executable, "-m", "esco_agent.web"], cwd=work,
                              env=server_env(port), stdout=log.open("w"), stderr=subprocess.STDOUT)
    procs.append(server)
    xvfb = subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", f"{SIZE[0]}x{SIZE[1]}x24",
                             "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(xvfb)
    raw = OUT / "raw.mkv"
    marks: list[dict] = []
    info: dict = {}
    ok = False
    t0 = None
    try:
        wait_up(url, server, log)
        with sync_playwright() as p:
            ctx = browser(p, work / "profile")
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            def mark(name: str) -> None:
                marks.append({"event": name, "t": round(time.time() - t0, 2)})
                print(f"{marks[-1]['t']:7.1f}s  {name}", flush=True)

            def failed() -> str | None:
                el = page.locator("#toast")
                return el.text_content() if el.is_visible() else None

            def reply(name: str) -> None:
                """Send what is in the box and wait for the whole reply."""
                mark(f"send_{name}")
                page.keyboard.press("Enter")
                page.wait_for_function(
                    "() => !document.getElementById('send').disabled"
                    " && !document.querySelector('.msg.typing .spin, .msg.streaming')",
                    timeout=REPLY_TIMEOUT_MS)
                mark(f"done_{name}")
                if failed():
                    raise TakeFailed(f"The app showed an error: {failed()}")

            def ask(text: str, name: str) -> None:
                page.click("#box")
                page.keyboard.type(text, delay=TYPE_DELAY_MS)
                page.wait_for_timeout(500)
                reply(name)

            def last_reply() -> str:
                return (page.locator(".msg.assistant .bubble").last.text_content() or "").strip()

            # the study's start screen: filled in before recording starts
            page.goto(url)
            page.click("#langs button[data-lang=en]")
            page.fill("#email", "demo@example.com")
            page.fill("#uc-name", COURSE_NAME)
            page.fill("#expected", "(demo)")
            page.click("#start")
            page.wait_for_selector("#s2.on")
            page.wait_for_timeout(2500)          # the greeting types itself out

            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "60",
                 "-video_size", f"{SIZE[0]}x{SIZE[1]}", "-draw_mouse", "0", "-i", DISPLAY,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                 str(raw)], stdin=subprocess.PIPE, env={**os.environ, "DISPLAY": DISPLAY})
            procs.append(ff)
            t0 = time.time()
            page.wait_for_timeout(1500)
            mark("chat")

            # 1. paste the course page; open one skill to show its evidence
            page.fill("#box", course)
            page.wait_for_timeout(2000)
            reply("paste")
            page.wait_for_timeout(3000)
            page.locator("#panel details.skill summary").first.click()
            mark("evidence")
            page.wait_for_timeout(5000)

            # 2. the teacher decides: keep the list, then remove one skill with a reason
            ask(KEEP, "keep")
            page.wait_for_timeout(2000)
            cards = page.locator("#panel details.skill.accepted")
            if not cards.count():
                raise TakeFailed("Nothing was kept. The assistant said:\n" + last_reply())
            n = cards.last.get_attribute("data-n")   # the last one: usually a generic skill
            card = page.locator(f'#panel details.skill[data-n="{n}"]')
            info["removed"] = card.locator(".label").text_content()
            card.locator("summary").click()
            card.scroll_into_view_if_needed()
            page.wait_for_timeout(1500)
            mark("remove")
            card.locator('[data-act="reject-ask"]').click()
            # bring the four explained reasons fully into view, then let them be read
            page.evaluate("el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })",
                          card.locator(".reasons").element_handle())
            page.wait_for_timeout(4500)
            card.locator('[data-act="other-ask"]').click()
            page.wait_for_timeout(600)
            page.keyboard.type(REMOVE_NOTE, delay=TYPE_DELAY_MS)
            page.wait_for_timeout(800)
            card.locator('.reasons [data-act="reject"]').click()
            # removed skills go into the collapsed "Removed" group: present, not visible
            page.wait_for_selector(f'#panel details.skill.rejected[data-n="{n}"]', state="attached")
            page.wait_for_timeout(2500)

            # 3. occupations, then what the course lacks for one of them
            ask(ASK_OCC, "occ")
            labels = [x.strip() for x in page.locator("#panel details.occ .label").all_text_contents()]
            if not labels:
                raise TakeFailed("No occupations appeared. The assistant said:\n" + last_reply())
            page.wait_for_timeout(4000)
            occ = next((x for x in labels if x.lower() == a.occupation.lower()), labels[0])
            info["occupation"] = occ
            ask(ASK_LACK.format(occ=occ), "lack")
            missing = page.evaluate("""occ => {
                for (const d of document.querySelectorAll('#panel details.occ')) {
                  if (d.querySelector('.label').textContent.trim() !== occ) continue;
                  return [...d.querySelectorAll('li')].filter(li => !li.querySelector('button'))
                                                       .map(li => li.textContent.trim());
                }
                return []; }""", occ)
            page.wait_for_timeout(5000)
            skill = a.skill or (missing[0] if missing else None)
            if not skill:
                raise TakeFailed(f"The course lacks nothing for {occ}. Try --occupation.")
            info["skill"] = skill

            # 4. text for a missing skill, checked; accept it, and the re-check runs
            ask(ASK_DRAFT.format(skill=skill), "draft")
            page.wait_for_timeout(6000)
            if not page.locator("#panel details.change.st-proposed").count():
                raise TakeFailed("No text change to accept. The assistant said:\n" + last_reply()
                                 + "\nTry --skill with another missing skill: " + ", ".join(missing))
            ask(ACCEPT, "accept")
            page.wait_for_timeout(4000)
            if not page.locator("#panel details.change.st-accepted").count():
                raise TakeFailed("The change was not accepted. The assistant said:\n" + last_reply())

            # 5. end on the accepted change in the panel
            change = page.locator("#panel details.change.st-accepted").first
            page.evaluate("el => el.scrollIntoView({ behavior: 'smooth', block: 'start' })",
                          change.element_handle())
            if change.get_attribute("open") is None:
                change.locator("summary").click()
            page.wait_for_timeout(7000)
            mark("end")
            ok = True
            ctx.close()
    except TakeFailed as e:
        print(f"\nThis take can't be used: {e}")
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                if proc.stdin:
                    try:
                        proc.communicate(b"q", timeout=30)    # ffmpeg: finish the file
                    except Exception:
                        proc.kill()
                else:
                    proc.terminate()
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        (OUT / "timeline.json").write_text(json.dumps(
            {"marks": marks, "course": COURSE_NAME, "source": SOURCE, **info,
             "complete": ok}, indent=2))
        shutil.copy(log, OUT / "server.log")
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'Done' if ok else 'Incomplete'}: {raw}, {OUT / 'timeline.json'}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
