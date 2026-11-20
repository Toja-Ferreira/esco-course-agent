"""Edit the recorded demo with ffmpeg: speed up the waits for the model (with a
"sped up" label), add a caption bar under the page with one caption per step,
and export demo/out/demo.mp4 (H.264, 60 fps) and demo/out/demo.gif (step 1, for
the README). Sizes follow the recording (1080p from record.py).

    python demo/edit.py            # after demo/record.py
Also writes demo/out/frames/*.png, one per caption, to check the result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
FONT_CANDIDATES = ["/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
                   "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
BADGE_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # has the arrow glyph
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
MIN_SPEED = 4.0          # waits for the model run at least this much faster
MAX_WAIT_S = 5.0         # and last at most this long on screen
MIN_FAST_S = 4.0         # shorter waits play at normal speed (a label would only flash)
LEAD_IN_S = 0.3          # kept before the first event
BAR = 96                 # caption bar under the page (at 720p; scaled with the video)
CREDIT = ("Course text adapted from “Learn to code for data analysis”, "
          "The Open University (OpenLearn), CC BY-NC-SA")
CAPTIONS = [  # (from event, to event, text)
    ("chat", "evidence", "Finds ESCO skills in the course text, with evidence"),
    ("evidence", "send_keep", "Every skill comes from ESCO, quoted from the text"),
    ("send_keep", "send_occ", "The teacher decides: keep skills, or remove them with a reason"),
    ("send_occ", "send_draft", "Ranks ESCO occupations and checks what the course lacks"),
    ("send_draft", "send_accept", "Suggests text for a missing skill, checked before it is shown"),
    ("send_accept", "end", "Accepted text is re-checked for what else it shows"),
]
WAITS = ["paste", "keep", "occ", "lack", "draft", "accept"]


def font() -> str:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return f
    sys.exit("No font found; edit FONT_CANDIDATES.")


def segments(t: dict) -> list[tuple[float, float, float]]:
    """(start, end, speed) in the raw video, in order."""
    waits = []
    for w in WAITS:
        if f"send_{w}" in t and f"done_{w}" in t:
            waits.append((t[f"send_{w}"], t[f"done_{w}"]))
    out, pos = [], t["chat"] - LEAD_IN_S
    for a, b in waits:
        if a > pos:
            out.append((pos, a, 1.0))
        speed = 1.0 if b - a < MIN_FAST_S else max(MIN_SPEED, (b - a) / MAX_WAIT_S)
        out.append((a, b, speed))
        pos = b
    out.append((pos, t["end"] + 0.5, 1.0))
    return out


def to_out(segs, t_raw: float) -> float:
    """Where a moment of the raw video lands in the edited one."""
    acc = 0.0
    for a, b, s in segs:
        if t_raw <= b:
            return acc + max(0.0, t_raw - a) / s
        acc += (b - a) / s
    return acc


def esc(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(OUT / "raw.mkv"))
    ap.add_argument("--timeline", default=str(OUT / "timeline.json"))
    ap.add_argument("--gif-seconds", type=float, default=14.0)
    a = ap.parse_args()
    t = {m["event"]: m["t"] for m in json.loads(Path(a.timeline).read_text())["marks"]}
    for need in ("chat", "end"):
        if need not in t:
            sys.exit(f"The timeline has no '{need}' event; was the recording complete?")
    segs = segments(t)
    total = to_out(segs, t["end"] + 0.5)
    f = font()
    badge = next((b for b in BADGE_FONTS if Path(b).exists()), f)
    height = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=height", "-of", "csv=p=0", a.raw],
                                capture_output=True, text=True, check=True).stdout.strip())
    zoom = height / 720                    # sizes below are for 720p
    bar = int(BAR * zoom) // 2 * 2
    px = lambda v: int(round(v * zoom))
    tmp = Path(tempfile.mkdtemp(prefix="esco-edit-"))
    OUT.mkdir(exist_ok=True)

    # one input per segment (seeking in the file), so ffmpeg never buffers the whole video
    inputs = []
    for a_, b, _ in segs:
        inputs += ["-ss", f"{a_:.3f}", "-t", f"{b - a_:.3f}", "-i", a.raw]
    chains = [f"[{i}:v]setpts=(PTS-STARTPTS)/{s:.3f}[s{i}]" for i, (_, _, s) in enumerate(segs)]
    chains.append("".join(f"[s{i}]" for i in range(len(segs)))
                  + f"concat=n={len(segs)}:v=1:a=0,fps=60,"
                  f"pad=w=iw:h=ih+{bar}:x=0:y=0:color=0x16181D[cat]")

    draw, shots = [], []
    for k, (ev_a, ev_b, text) in enumerate(CAPTIONS):
        if ev_a not in t:
            continue
        start = to_out(segs, t[ev_a])
        end = to_out(segs, t[ev_b]) if ev_b in t else total
        tf = tmp / f"cap{k}.txt"
        tf.write_text(text)
        draw.append(f"drawtext=fontfile='{esc(Path(f))}':textfile='{esc(tf)}':fontcolor=white:fontsize={px(28)}"
                    f":x={px(28)}:y=h-{bar}+{px(18)}:enable='between(t,{start:.2f},{end:.2f})'")
        shots.append((k, (start + end) / 2))
    for i, (a_, b, s) in enumerate(segs):
        if s > 1:
            start, end = to_out(segs, a_), to_out(segs, b)
            tf = tmp / f"fast{i}.txt"
            tf.write_text("\u25B6\u25B6")             # fast-forward, while the wait plays faster
            draw.append(f"drawtext=fontfile='{esc(Path(badge))}':textfile='{esc(tf)}':fontsize={px(24)}"
                        f":box=1:boxcolor=0xC0392B@0.9:boxborderw={px(10)}:fontcolor=white"
                        f":x=w-text_w-{px(28)}:y=h-{bar}+{px(16)}:enable='between(t,{start:.2f},{end:.2f})'")
    cf = tmp / "credit.txt"
    cf.write_text(CREDIT)
    draw.append(f"drawtext=fontfile='{esc(Path(f))}':textfile='{esc(cf)}':fontsize={px(16)}"
                f":fontcolor=0xA9B0BC:x={px(28)}:y=h-{px(30)}")
    chains.append("[cat]" + ",".join(draw) + "[v]")

    mp4 = OUT / "demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", ";".join(chains), "-map", "[v]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)

    gif_end = min(a.gif_seconds, to_out(segs, t.get("send_keep", t["end"])))
    gif = OUT / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", f"{gif_end:.2f}", "-i", str(mp4),
                    "-vf", "fps=12,scale=960:-1:flags=lanczos,split[a][b];"
                           "[a]palettegen=max_colors=128:stats_mode=diff[p];"
                           "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)], check=True)

    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)
    for k, at in shots + [("fast", to_out(segs, (t.get("send_paste", 0) + t.get("done_paste", 0)) / 2)),
                          ("first", 1.0), ("last", total - 1.0)]:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(mp4),
                        "-frames:v", "1", str(frames / f"{k}.png")], check=True)

    mb = mp4.stat().st_size / 1e6
    print(f"demo.mp4: {total:.1f} s, {mb:.1f} MB; demo.gif: {gif_end:.1f} s, "
          f"{gif.stat().st_size / 1e6:.1f} MB")
    for a_, b, s in segs:
        if s > 1:
            print(f"  wait {b - a_:5.1f} s shown in {(b - a_) / s:4.1f} s ({s:.0f}x)")
    if mb > 50:
        print("Over 50 MB: raise the CRF (18) in the encode step for a smaller file.")


if __name__ == "__main__":
    main()
