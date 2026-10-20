#!/usr/bin/env python3
"""
Efficient ground-truth annotation tool (pupil center + glints).

Reads gt_builder/sample_manifest.csv and lets you annotate each frame with a
magnified, sub-pixel-accurate view. Output is a JSON ground truth that is a
SUPERSET of ground_truth/ground_truth.json, so it plugs straight into the
evaluator (accuracy_eval/eval_vs_gt.py with --gt pointing here).

Per frame you provide:
  - pupil center      (left-click on the zoomed view)
  - glint(s)          ('g' to enter glint mode; 1 click for bright, 2 for dark)
  - or mark a blink   ('b')  / skip ('s')

Controls
--------
  left-click   place pupil center (or glint, in glint mode)
  g            toggle glint annotation mode
  b            mark frame as BLINK (no pupil)  -> saved, advance
  Enter / n    accept & next   (requires a pupil center unless blink)
  p            previous frame
  u            undo last click
  z / x        zoom out / in around cursor
  s            skip (not annotated)  -> advance
  q / Esc      save & quit

Resume-safe: already-annotated frames are loaded and can be revisited; quitting
saves incrementally. A SHA-256 hash + metadata file is written next to the JSON.

NOTE: this is a GUI tool (needs a display); run it on your local machine:
      python3 gt_builder/annotate.py
"""
import os
import sys
import csv
import json
import time
import hashlib
import datetime
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "sample_manifest.csv")
OUT_JSON = os.path.join(HERE, "ground_truth_large.json")
ANNOTATOR = os.environ.get("USER", "harre")

WIN = "GT annotator  [click=pupil | g=glint | b=blink | enter=next | p=prev | u=undo | z/x=zoom | s=skip | q=quit]"
ZOOM_DEFAULT = 3.0
CROP = 240   # half-size of the region cropped around the seed (full-res px)


def load_manifest():
    rows = []
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_existing():
    if os.path.exists(OUT_JSON):
        return json.load(open(OUT_JSON))
    return {}


def save_json(data):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT_JSON)
    h = hashlib.sha256(open(OUT_JSON, "rb").read()).hexdigest()
    with open(OUT_JSON.replace(".json", "_hash.txt"), "w") as f:
        f.write(f"{h}  {os.path.basename(OUT_JSON)}\n")
        f.write(f"generated: {datetime.datetime.utcnow().isoformat()}Z\n")
        f.write(f"N_frames:  {len(data)}\n")


class Annotator:
    def __init__(self, rows, data):
        self.rows = rows
        self.data = data
        self.i = 0
        self.zoom = ZOOM_DEFAULT
        self.reset_frame_state()

    def reset_frame_state(self):
        self.pupil = None        # (x,y) full-res
        self.glints = []         # list of (x,y) full-res
        self.glint_mode = False
        self.is_blink = False
        self.cursor = (0, 0)     # in window coords
        self.cx0, self.cy0 = 0, 0  # crop origin in full-res

    # ---- coordinate mapping ----
    def win_to_full(self, wx, wy):
        return self.cx0 + wx / self.zoom, self.cy0 + wy / self.zoom

    def full_to_win(self, fx, fy):
        return int(round((fx - self.cx0) * self.zoom)), int(round((fy - self.cy0) * self.zoom))

    def current(self):
        return self.rows[self.i]

    def load_frame(self):
        r = self.current()
        key = f"{r['participant_id']}_{r['session']}_{r['file_name']}"
        img = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        self.reset_frame_state()
        # seed crop center: existing annotation > manifest auto seed > image center
        if key in self.data and self.data[key].get("pupil_gt"):
            sx, sy = self.data[key]["pupil_gt"]
            self.pupil = (float(sx), float(sy))
            self.glints = [tuple(g) for g in self.data[key].get("glints", [])]
            self.is_blink = bool(self.data[key].get("is_blink"))
        else:
            ax, ay = float(r.get("auto_cx", -1)), float(r.get("auto_cy", -1))
            if ax > 0:
                self.seed = (ax, ay)
            else:
                self.seed = (img.shape[1] / 2, img.shape[0] / 2)
        seed = self.pupil if self.pupil else getattr(self, "seed", (img.shape[1] / 2, img.shape[0] / 2))
        self.cx0 = int(max(0, min(img.shape[1] - 2 * CROP, seed[0] - CROP)))
        self.cy0 = int(max(0, min(img.shape[0] - 2 * CROP, seed[1] - CROP)))
        self.img = img
        return key

    def render(self):
        r = self.current()
        crop = self.img[self.cy0:self.cy0 + 2 * CROP, self.cx0:self.cx0 + 2 * CROP]
        view = cv2.resize(crop, (0, 0), fx=self.zoom, fy=self.zoom,
                          interpolation=cv2.INTER_NEAREST)
        # draw pupil
        if self.pupil:
            x, y = self.full_to_win(*self.pupil)
            cv2.drawMarker(view, (x, y), (255, 80, 80), cv2.MARKER_CROSS, 26, 1)
            cv2.circle(view, (x, y), 2, (255, 80, 80), -1)
        # draw glints
        for gx, gy in self.glints:
            x, y = self.full_to_win(gx, gy)
            cv2.drawMarker(view, (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 1)
        # crosshair magnifier at cursor
        cx, cy = self.cursor
        cv2.line(view, (cx, 0), (cx, view.shape[0]), (60, 200, 60), 1)
        cv2.line(view, (0, cy), (view.shape[1], cy), (60, 200, 60), 1)
        # HUD
        done = sum(1 for k in self.data if self.data[k].get("pupil_gt") or self.data[k].get("is_blink"))
        hud = [
            f"[{self.i+1}/{len(self.rows)}] {r['participant_id']} {r['session']} {r['file_name']}  type={r['frame_type']}",
            f"annotated={done}  mode={'GLINT' if self.glint_mode else 'PUPIL'}  "
            f"pupil={'OK' if self.pupil else '--'}  glints={len(self.glints)}  "
            f"{'BLINK' if self.is_blink else ''}",
        ]
        bar = np.zeros((48, view.shape[1], 3), np.uint8)
        for j, t in enumerate(hud):
            cv2.putText(bar, t, (6, 18 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(WIN, np.vstack([bar, view]))

    def on_mouse(self, event, x, y, flags, param):
        self.cursor = (x, max(0, y - 48))  # account for HUD bar height
        fy = self.cursor[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            fx_full, fy_full = self.win_to_full(x, fy)
            if self.glint_mode:
                maxg = 1 if self.current()["frame_type"] == "bright" else 2
                self.glints.append((round(fx_full, 2), round(fy_full, 2)))
                if len(self.glints) > maxg:
                    self.glints = self.glints[-maxg:]
            else:
                self.pupil = (round(fx_full, 2), round(fy_full, 2))
                self.is_blink = False
            self.render()

    def commit(self):
        r = self.current()
        key = f"{r['participant_id']}_{r['session']}_{r['file_name']}"
        if self.is_blink:
            self.data[key] = {
                "participant_id": r["participant_id"], "session": r["session"],
                "frame_type": r["frame_type"], "pupil_gt": None, "glints": [],
                "is_blink": True, "annotator": ANNOTATOR,
                "timestamp_iso": datetime.datetime.utcnow().isoformat() + "Z",
            }
            return True
        if not self.pupil:
            return False
        self.data[key] = {
            "participant_id": r["participant_id"], "session": r["session"],
            "frame_type": r["frame_type"],
            "pupil_gt": [self.pupil[0], self.pupil[1]],
            "glints": [list(g) for g in self.glints],
            "is_blink": False, "annotator": ANNOTATOR,
            "timestamp_iso": datetime.datetime.utcnow().isoformat() + "Z",
        }
        return True

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load_frame()
        while True:
            self.render()
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k in (13, ord('n')):           # enter / next
                if self.commit():
                    save_json(self.data)
                    if self.i < len(self.rows) - 1:
                        self.i += 1
                        self.load_frame()
                    else:
                        print("Reached last frame.")
                else:
                    print("[!] Need a pupil center (or press 'b' for blink) before advancing.")
            elif k == ord('p'):                 # previous
                if self.i > 0:
                    self.i -= 1
                    self.load_frame()
            elif k == ord('g'):
                self.glint_mode = not self.glint_mode
            elif k == ord('b'):
                self.is_blink = True
                self.pupil = None
                self.glints = []
            elif k == ord('u'):
                if self.glint_mode and self.glints:
                    self.glints.pop()
                else:
                    self.pupil = None
            elif k == ord('s'):                 # skip
                if self.i < len(self.rows) - 1:
                    self.i += 1
                    self.load_frame()
            elif k == ord('z'):
                self.zoom = max(1.0, self.zoom - 0.5)
            elif k == ord('x'):
                self.zoom = min(8.0, self.zoom + 0.5)
        save_json(self.data)
        cv2.destroyAllWindows()
        print(f"Saved {len(self.data)} annotations to {OUT_JSON}")


def main():
    if not os.path.exists(MANIFEST):
        sys.exit(f"Manifest not found: {MANIFEST}\nRun sample_frames.py first.")
    rows = load_manifest()
    data = load_existing()
    # jump to first un-annotated frame
    ann = Annotator(rows, data)
    for i, r in enumerate(rows):
        key = f"{r['participant_id']}_{r['session']}_{r['file_name']}"
        if key not in data:
            ann.i = i
            break
    ann.run()


if __name__ == "__main__":
    main()
