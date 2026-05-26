# 아래 코드를 터미널에서 실행할 것
# opencv 설치
# C:\Anaconda\python.exe -m pip install opencv-python
# 설치가 완료되었거나, 이미 설치되어 있다면, 아래 명령어로 ROI 설정 GUI를 실행할 수 있음
# python make_ROI_from_videos.py
# **Shelter 좌표는 시계방향으로 찍을 것**
# 오류 발생 시 설명 참조

import os
import json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2

def _pick_roi_paths():
    result = {}
    root = tk.Tk()
    root.title("ROI 설정 경로")
    root.attributes("-topmost", True)
    root.configure(bg='#F5F5F5')
    root.resizable(False, False)

    vid_var = tk.StringVar(value='')
    out_var = tk.StringVar(value='')

    frm = tk.LabelFrame(root, text=" 경로 설정 ", bg='#F5F5F5',
                         font=('Arial', 10, 'bold'), padx=10, pady=6)
    frm.grid(row=0, column=0, padx=15, pady=(12, 4), sticky='ew')

    def _browse_file(var):
        p = filedialog.askopenfilename(parent=root, title="영상 파일 선택",
                                       filetypes=[("동영상", "*.avi *.mp4 *.mov"), ("All", "*.*")])
        if p: var.set(p)

    def _browse_save(var):
        folder = filedialog.askdirectory(parent=root, title="roi_info.json 저장 폴더")
        if folder: var.set(os.path.join(folder, "roi_info.json"))

    for i, (label, var, btn_fn) in enumerate([
        ("영상 파일", vid_var, _browse_file),
        ("저장 경로", out_var, _browse_save),
    ]):
        tk.Label(frm, text=label, bg='#F5F5F5', width=10, anchor='e',
                 font=('Arial', 9)).grid(row=i, column=0, padx=(4, 6), pady=4, sticky='e')
        tk.Entry(frm, textvariable=var, width=65,
                 font=('Consolas', 8)).grid(row=i, column=1, pady=4, sticky='ew')
        tk.Button(frm, text="찾기", bg='#E0E0E0',
                  command=lambda v=var, f=btn_fn: f(v)
                  ).grid(row=i, column=2, padx=(4, 6), pady=4)

    def _confirm():
        result['video_path']  = vid_var.get()
        result['output_path'] = out_var.get()
        root.destroy()

    tk.Button(root, text="  ROI setting start  ", command=_confirm,
              bg='#2B7BBD', fg='white', font=('Arial', 10, 'bold'),
              relief='flat', padx=10, pady=6
              ).grid(row=1, column=0, pady=12)

    root.update_idletasks()
    w, h   = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
    root.mainloop()
    return result


def _collect_points(frame, message, n_points, hint=''):
    pts = []

    def _redraw():
        d = frame.copy()
        cv2.putText(d, message, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(d, f"{len(pts)}/{n_points}  (Right-click=Undo / ESC=Cancel)",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if hint:
            cv2.putText(d, hint, (10, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        for j, p in enumerate(pts):
            cv2.circle(d, p, 6, (0, 255, 0), -1)
            cv2.putText(d, str(j + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("ROI setting", d)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n_points:
            pts.append((x, y))
            _redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            _redraw()

    cv2.setMouseCallback("ROI setting", click_event)
    _redraw()

    while len(pts) < n_points:
        key = cv2.waitKey(50)
        if key == 27:
            raise RuntimeError("ROI setting canceled (ESC)")

    print(f"  → {n_points} points completed")
    return pts


# ── 경로 선택 GUI ──
_roi_cfg        = _pick_roi_paths()
VIDEO_PATH_ROI  = _roi_cfg.get('video_path',  '')
OUTPUT_PATH_ROI = _roi_cfg.get('output_path', '')

if not VIDEO_PATH_ROI or not OUTPUT_PATH_ROI:
    print("⚠ Path not set → Skip ROI setting")
else:
    cap   = cv2.VideoCapture(VIDEO_PATH_ROI)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ I can't read the video.")
    else:
        cv2.namedWindow("ROI setting", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI setting", 960, 720)
        roi_data = {}

        # Step 1 — Arena 원주 8점
        print("\nStep 1: Click Arena circumference 8 times")
        pts = _collect_points(frame, "Step1: Arena Click circumference 8 times", 8)
        (cx, cy), radius = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
        roi_data['arena_center']      = [float(cx), float(cy)]
        roi_data['arena_radius']      = float(radius)
        roi_data['arena_diameter_cm'] = 100.0
        roi_data['px_per_cm']         = float(radius * 2 / 100.0)
        print(f"  Arena center ({cx:.1f}, {cy:.1f}), 반지름 {radius:.1f} px")
        cv2.circle(frame, (int(cx), int(cy)), int(radius), (255, 100, 0), 2)

        # Step 2 — Shelter 모서리 4점
        print("\nStep 2: Click the 4 corners of the Shelter")
        pts = _collect_points(frame, "Step2: Click the 4 corners of the Shelter", 4,
                              hint="* Click in clockwise order (e.g. top-left -> top-right -> bottom-right -> bottom-left)")
        (sx, sy), _ = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
        roi_data['shelter_points'] = pts
        roi_data['shelter_center'] = [float(sx), float(sy)]
        print(f"  Shelter Center ({sx:.1f}, {sy:.1f})")
        cv2.polylines(frame, [np.array(pts, np.int32).reshape(-1, 1, 2)], True, (0, 200, 255), 2)

        # Step 3 — Landmark 1 원주 4점 (선택, ESC=건너뜀)
        print("\nStep 3: Click Landmark 1 circumference 4 times  (ESC = skip)")
        try:
            pts = _collect_points(frame, "Step3: Landmark 1  (ESC=Skip)", 4)
            (lm1x, lm1y), lm1r = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
            roi_data['landmark1_center'] = [float(lm1x), float(lm1y)]
            roi_data['landmark1_radius'] = float(lm1r)
            print(f"  Landmark1 ({lm1x:.1f}, {lm1y:.1f}), 반지름 {lm1r:.1f} px")
            cv2.circle(frame, (int(lm1x), int(lm1y)), int(lm1r), (0, 165, 255), 2)

            # Step 4 — Landmark 2 원주 4점 (선택, ESC=건너뜀)
            print("\nStep 4: Click Landmark 2 circumference 4 times  (ESC = skip)")
            try:
                pts = _collect_points(frame, "Step4: Landmark 2  (ESC=Skip)", 4)
                (lm2x, lm2y), lm2r = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
                roi_data['landmark2_center'] = [float(lm2x), float(lm2y)]
                roi_data['landmark2_radius'] = float(lm2r)
                print(f"  Landmark2 ({lm2x:.1f}, {lm2y:.1f}), 반지름 {lm2r:.1f} px")
                cv2.circle(frame, (int(lm2x), int(lm2y)), int(lm2r), (0, 165, 255), 2)
            except RuntimeError:
                print("  → Landmark 2 skipped")
        except RuntimeError:
            print("  → Landmark 1 & 2 skipped")

        cv2.destroyAllWindows()

        save_dir = os.path.dirname(OUTPUT_PATH_ROI)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(OUTPUT_PATH_ROI, 'w', encoding='utf-8') as f:
            json.dump(roi_data, f, indent=4)

        print(f"\n✅ ROI 저장 완료: {OUTPUT_PATH_ROI}")
        print(json.dumps(roi_data, indent=2, ensure_ascii=False))