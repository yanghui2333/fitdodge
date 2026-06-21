"""Settings UI: video import, pose parsing, track management."""

import json
import os
import shutil
import threading
import tkinter.filedialog
import tkinter
import cv2
import numpy as np
import pygame


TRACKS_DIR = os.path.join(os.path.dirname(__file__), "tracks")


def get_tracks():
    """Return list of saved tracks: [(name, path, info_dict), ...]"""
    if not os.path.exists(TRACKS_DIR):
        return []
    tracks = []
    for fname in sorted(os.listdir(TRACKS_DIR)):
        if fname.endswith(".json"):
            path = os.path.join(TRACKS_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                name = data.get("name", fname.replace(".json", ""))
                tracks.append((name, path, data))
            except:
                pass
    return tracks


def delete_track(path):
    try:
        os.remove(path)
        return True
    except:
        return False


def _pick_video_file():
    root = tkinter.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = tkinter.filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
    root.destroy()
    return path if path else ""


def _parse_video_async(video_path, callback):
    """Parse video in background, callback(progress, done, status_text, result_data)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    LM_NAMES = [
        "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
        "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT",
        "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
        "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY",
        "LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB",
        "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
        "LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL",
        "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
    ]
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            callback(0, True, "Cannot open video", None); return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_v2.task")
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts, running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        poses = []
        sample_every = max(1, int(fps / 10))
        ts = 0
        for fi in range(total):
            ret, frame = cap.read()
            if not ret: break
            if fi % sample_every != 0: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(fi * 1000 / fps)
            result = landmarker.detect_for_video(mp_img, ts)
            if result.pose_landmarks:
                lm_dict = {}
                for i, lm in enumerate(result.pose_landmarks[0]):
                    if i < len(LM_NAMES):
                        lm_dict[LM_NAMES[i]] = {"x":lm.x,"y":lm.y,"z":lm.z,"v":lm.visibility}
                poses.append({"frame":fi,"time_ms":ts,"landmarks":lm_dict})
            if fi % 30 == 0:
                callback((fi+1)/total, False, f"Parsing... {int((fi+1)/total*100)}%", None)
        landmarker.close(); cap.release()
        if poses:
            # Extract key poses
            key_poses = [poses[0]]
            last_frame = poses[0]["frame"]
            KJ = ["LEFT_WRIST","RIGHT_WRIST","LEFT_ELBOW","RIGHT_ELBOW"]
            for p in poses[1:]:
                if p["frame"] - last_frame < 5: continue
                prev_lm = key_poses[-1]["landmarks"]; curr_lm = p["landmarks"]
                td, cnt = 0.0, 0
                for n in KJ:
                    if n in prev_lm and n in curr_lm:
                        td += abs(prev_lm[n]["x"]-curr_lm[n]["x"])+abs(prev_lm[n]["y"]-curr_lm[n]["y"]); cnt += 1
                if cnt > 0 and td/cnt > 0.06:
                    key_poses.append(p); last_frame = p["frame"]
            data = {"name": os.path.splitext(os.path.basename(video_path))[0],
                    "source": video_path, "total_frames": total,
                    "sampled_frames": len(poses), "key_poses": len(key_poses),
                    "poses": key_poses}
            callback(1.0, True, f"Done! {len(key_poses)} key poses", data)
        else:
            callback(1.0, True, "No poses detected", None)
    except Exception as e:
        callback(0, True, f"Error: {e}", None)


def run_settings(screen, cap, pose_detector, fonts):
    """Settings screen with track management."""
    W, H = screen.get_width(), screen.get_height()
    mid_w, mid_h = W//2, H//2
    input_path = ""; input_active = False
    status_text = ""; parsing = False
    preview_idx = 0; parsed_data = None; scroll_timer = 0
    tracks = get_tracks(); selected_track = None
    delete_confirm = None  # (track_path, timer)
    font_lg = pygame.font.Font(None, max(28, H//24))
    font_md = pygame.font.Font(None, max(22, H//30))
    font_sm = pygame.font.Font(None, max(16, H//40))
    font_xs = pygame.font.Font(None, max(14, H//45))
    CONNECTIONS = [(11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),
                   (23,25),(25,27),(24,26),(26,28)]
    LM_NAMES = ["","","","","","","","","","","",
                "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
                "LEFT_WRIST","RIGHT_WRIST","","","","","","",
                "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
                "LEFT_ANKLE","RIGHT_ANKLE","","","",""]
    # Named landmarks for skeleton drawing
    LM_NAMES_D = {11:"LEFT_SHOULDER",12:"RIGHT_SHOULDER",13:"LEFT_ELBOW",14:"RIGHT_ELBOW",
                  15:"LEFT_WRIST",16:"RIGHT_WRIST",23:"LEFT_HIP",24:"RIGHT_HIP",
                  25:"LEFT_KNEE",26:"RIGHT_KNEE",27:"LEFT_ANKLE",28:"RIGHT_ANKLE"}
    LM_NAMES_D = {
        0:"NOSE",1:"LEFT_EYE_INNER",2:"LEFT_EYE",3:"LEFT_EYE_OUTER",
        4:"RIGHT_EYE_INNER",5:"RIGHT_EYE",6:"RIGHT_EYE_OUTER",
        7:"LEFT_EAR",8:"RIGHT_EAR",9:"MOUTH_LEFT",10:"MOUTH_RIGHT",
        11:"LEFT_SHOULDER",12:"RIGHT_SHOULDER",13:"LEFT_ELBOW",14:"RIGHT_ELBOW",
        15:"LEFT_WRIST",16:"RIGHT_WRIST",
        23:"LEFT_HIP",24:"RIGHT_HIP",25:"LEFT_KNEE",26:"RIGHT_KNEE",
        27:"LEFT_ANKLE",28:"RIGHT_ANKLE",
    }
    clock = pygame.time.Clock()

    def on_progress(progress, done, msg, data):
        nonlocal parsing, status_text, parsed_data, tracks
        status_text = msg
        if done:
            parsing = False
            if data:
                # Save to tracks
                os.makedirs(TRACKS_DIR, exist_ok=True)
                name = data["name"]
                fname = f"{name}.json"
                path = os.path.join(TRACKS_DIR, fname)
                with open(path, "w") as f:
                    json.dump(data, f)
                tracks = get_tracks()
                parsed_data = data
                status_text = f"Saved: {name}"

    while True:
        mx, my = pygame.mouse.get_pos()
        click = False
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return
                if input_active:
                    if event.key == pygame.K_BACKSPACE: input_path = input_path[:-1]
                    elif event.key == pygame.K_RETURN: input_active = False
                    elif event.unicode and event.unicode.isprintable(): input_path += event.unicode
                # Keyboard navigation for preview
                if parsed_data and "poses" in parsed_data:
                    total_poses = len(parsed_data["poses"])
                    if event.key == pygame.K_LEFT:
                        preview_idx = max(0, preview_idx - 1)
                    elif event.key == pygame.K_RIGHT:
                        preview_idx = min(total_poses - 1, preview_idx + 1)
            if event.type == pygame.MOUSEBUTTONDOWN: click = True

        # Background
        screen.fill((18,18,30))
        title = font_lg.render("Settings - Track Manager", True, (80,220,255))
        screen.blit(title, (mid_w - title.get_width()//2, int(H*0.04)))

        # Close X
        cx_r = int(W*0.025)
        close_rect = pygame.Rect(W - int(W*0.08), int(H*0.03), int(W*0.06), int(H*0.06))
        pygame.draw.rect(screen, (60,60,80), close_rect, border_radius=8)
        ct = font_md.render("X", True, (255,100,100))
        screen.blit(ct, (close_rect.centerx - ct.get_width()//2, close_rect.centery - ct.get_height()//2))
        if click and close_rect.collidepoint(mx, my): return

        # ── LEFT: Import panel ──
        lx, lw = int(W*0.04), int(W*0.38)
        py = int(H*0.15)
        screen.blit(font_sm.render("Import Video", True, (180,180,200)), (lx, py))
        input_rect = pygame.Rect(lx, py+int(H*0.04), lw, int(H*0.05))
        bc = (80,220,255) if input_active else (60,60,80)
        pygame.draw.rect(screen, (30,30,40), input_rect, border_radius=6)
        pygame.draw.rect(screen, bc, input_rect, width=2, border_radius=6)
        disp = input_path[-45:] if len(input_path)>45 else input_path
        if not disp and not input_active: disp = "Click to type or Browse..."
        it = font_sm.render(disp, True, (200,200,200) if input_path else (100,100,120))
        screen.blit(it, (input_rect.x+10, input_rect.centery - it.get_height()//2))
        if click and input_rect.collidepoint(mx, my): input_active = not input_active
        elif click: input_active = False
        # Buttons
        by2 = input_rect.bottom + int(H*0.025)
        bw2, bh2 = int(W*0.11), int(H*0.045)
        btn_browse = pygame.Rect(lx, by2, bw2, bh2)
        btn_parse = pygame.Rect(btn_browse.right+int(W*0.02), by2, bw2, bh2)
        bbc = (100,150,255) if btn_browse.collidepoint(mx,my) else (60,60,80)
        pygame.draw.rect(screen, bbc, btn_browse, border_radius=8)
        screen.blit(font_sm.render("Browse...", True, (255,255,255)),
                    (btn_browse.centerx - font_sm.render("Browse...",True,(255,255,255)).get_width()//2,
                     btn_browse.centery - font_sm.render("Browse...",True,(255,255,255)).get_height()//2))
        pbc = (80,220,120) if input_path and not parsing else (60,60,80)
        pygame.draw.rect(screen, pbc, btn_parse, border_radius=8)
        screen.blit(font_sm.render("Parse" if not parsing else "...", True, (255,255,255)),
                    (btn_parse.centerx - font_sm.render("Parse",True,(255,255,255)).get_width()//2,
                     btn_parse.centery - font_sm.render("Parse",True,(255,255,255)).get_height()//2))
        if status_text:
            screen.blit(font_xs.render(status_text, True, (200,200,100)), (lx, by2+bh2+int(H*0.02)))
        if click and btn_browse.collidepoint(mx,my):
            p = _pick_video_file()
            if p: input_path = p; status_text = f"Selected: {os.path.basename(p)}"
        if click and btn_parse.collidepoint(mx,my) and input_path and not parsing:
            if os.path.exists(input_path):
                parsing = True; status_text = "Parsing..."
                threading.Thread(target=_parse_video_async, args=(input_path, on_progress), daemon=True).start()
            else: status_text = "File not found!"

        # ── RIGHT: Track list ──
        rx, rw = int(W*0.46), int(W*0.50)
        screen.blit(font_sm.render("Saved Tracks", True, (180,180,200)), (rx, py))
        list_y = py + int(H*0.04)
        item_h = int(H*0.07)
        visible_items = (H - list_y - int(H*0.18)) // item_h

        if not tracks:
            screen.blit(font_md.render("No tracks yet", True, (100,100,120)), (rx+int(W*0.02), list_y+int(H*0.05)))
        else:
            for i, (name, path, data) in enumerate(tracks[:visible_items]):
                iy = list_y + i * item_h
                item_rect = pygame.Rect(rx, iy, rw, item_h-4)
                is_sel = (selected_track == path)
                ic = (50,50,70) if is_sel else (30,30,42)
                pygame.draw.rect(screen, ic, item_rect, border_radius=8)
                bc2 = (80,220,255) if is_sel else (50,50,65)
                pygame.draw.rect(screen, bc2, item_rect, width=2, border_radius=8)
                # Track name
                tn = font_sm.render(name, True, (255,255,255) if is_sel else (200,200,210))
                screen.blit(tn, (rx+int(W*0.02), iy+int(H*0.01)))
                # Info
                kp = data.get("key_poses", len(data.get("poses",[])))
                info = font_xs.render(f"{kp} poses", True, (150,150,170))
                screen.blit(info, (rx+int(W*0.02), iy+int(H*0.035)))
                # Preview button area
                pvw_rect = pygame.Rect(rx+rw-int(W*0.18), iy+4, int(W*0.08), item_h-8)
                ppc = (100,150,255) if pvw_rect.collidepoint(mx,my) else (60,60,80)
                pygame.draw.rect(screen, ppc, pvw_rect, border_radius=6)
                pvt = font_xs.render("Preview", True, (255,255,255))
                screen.blit(pvt, (pvw_rect.centerx-pvt.get_width()//2, pvw_rect.centery-pvt.get_height()//2))
                # Delete button
                del_rect = pygame.Rect(pvw_rect.right+int(W*0.01), iy+4, int(W*0.06), item_h-8)
                dc = (220,80,80) if del_rect.collidepoint(mx,my) else (60,60,80)
                pygame.draw.rect(screen, dc, del_rect, border_radius=6)
                dt2 = font_xs.render("Del", True, (255,255,255))
                screen.blit(dt2, (del_rect.centerx-dt2.get_width()//2, del_rect.centery-dt2.get_height()//2))
                # Click handling
                if click:
                    if item_rect.collidepoint(mx,my) and not pvw_rect.collidepoint(mx,my) and not del_rect.collidepoint(mx,my):
                        selected_track = path
                    if pvw_rect.collidepoint(mx,my):
                        selected_track = path
                        with open(path) as f: parsed_data = json.load(f)
                        preview_idx = 0
                    if del_rect.collidepoint(mx,my):
                        if delete_confirm and delete_confirm[0] == path:
                            delete_track(path); tracks = get_tracks(); selected_track = None; delete_confirm = None
                        else:
                            delete_confirm = (path, 60)

        # Delete confirmation
        if delete_confirm:
            dp, dtimer = delete_confirm
            if dtimer > 0:
                ctxt = font_xs.render("Click Delete again to confirm", True, (255,150,150))
                screen.blit(ctxt, (rx+int(W*0.02), list_y + visible_items*item_h + 4))
                delete_confirm = (dp, dtimer-1)

                # ── Preview panel ──
        if parsed_data and "poses" in parsed_data:
            poses = parsed_data["poses"]
            total = len(poses)
                    # ── Preview panel ──
        if parsed_data and "poses" in parsed_data:
            poses = parsed_data["poses"]
            total = len(poses)
            # -- LARGE skeleton (left, centered vertically) --
            BX, BY, BW, BH = lx, int(H*0.35), lw, int(H*0.86*H) - int(H*0.35) - 50
            # But BH must be computed in pixels:
            BX, BY, BW = lx, int(H*0.35), lw
            BH = int(H*0.86) - BY - 50
            pygame.draw.rect(screen, (25,25,38), (BX,BY,BW,BH), border_radius=10)
            pygame.draw.rect(screen, (50,50,65), (BX,BY,BW,BH), width=2, border_radius=10)
            screen.blit(font_sm.render("Pose "+str(preview_idx+1)+"/"+str(total), True, (80,220,255)), (BX+10, BY+6))
            lm = poses[preview_idx]["landmarks"]
            xs, ys = [], []
            for a_idx in (11,12,13,14,15,16,23,24,25,26,27,28):
                nm = LM_NAMES[a_idx] if a_idx < len(LM_NAMES) else ""
                if nm and nm in lm:
                    xs.append(lm[nm]["x"]); ys.append(lm[nm]["y"])
            if xs and ys:
                box_cx = BX + BW // 2; box_cy = BY + BH // 2
                lm_cx = (min(xs) + max(xs)) / 2; lm_cy = (min(ys) + max(ys)) / 2
                sc = min(BW*0.6/(max(xs)-min(xs) or 0.3), BH*0.6/(max(ys)-min(ys) or 0.5)) * 0.85
                def _ts(_x,_y): return (int(box_cx + (_x-lm_cx)*sc), int(box_cy + (_y-lm_cy)*sc))
            else:
                def _ts(_x,_y): return (int(BX+_x*BW*0.7), int(BY+_y*BH*0.7))
            for a,b in ((11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)):
                na = LM_NAMES[a] if a<len(LM_NAMES) else ""; nb = LM_NAMES[b] if b<len(LM_NAMES) else ""
                if na in lm and nb in lm:
                    ax, ay = _ts(lm[na]["x"], lm[na]["y"]); bx, by = _ts(lm[nb]["x"], lm[nb]["y"])
                    pygame.draw.line(screen, (100,180,255), (ax, ay), (bx, by), 3)
            for a_idx in (11,12,13,14,15,16,23,24,25,26,27,28):
                nm = LM_NAMES[a_idx] if a_idx < len(LM_NAMES) else ""
                if nm and nm in lm:
                    jx, jy = _ts(lm[nm]["x"], lm[nm]["y"])
                    pygame.draw.circle(screen, (0,200,255), (jx, jy), 6)
            # -- BOTTOM mini strip (full width) --
            PY, PH = int(H*0.86), int(H*0.10)
            MSX, MSW = lx, int(W*0.92)
            pygame.draw.rect(screen, (25,25,38), (MSX, PY, MSW, PH), border_radius=10)
            pygame.draw.rect(screen, (50,50,65), (MSX, PY, MSW, PH), width=2, border_radius=10)
            # (counter removed, shown on large skeleton instead)
            n_slots = min(11, total); half = n_slots // 2
            sp = max(0, preview_idx - half); ep = min(total-1, sp + n_slots - 1); sp = max(0, ep - n_slots + 1)
            for si in range(ep - sp + 1):
                pidx = sp + si; lm2 = poses[pidx]["landmarks"]; sw = MSW // (ep - sp + 1); sx = MSX + si * sw
                cur = (pidx == preview_idx)
                if cur: pygame.draw.rect(screen, (255,200,60), (sx, int(PY+8), sw-2, int(PH*0.65)), 2, border_radius=3)
                for a,b in ((11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)):
                    na = LM_NAMES[a] if a<len(LM_NAMES) else ""; nb = LM_NAMES[b] if b<len(LM_NAMES) else ""
                    if na in lm2 and nb in lm2:
                        ax = int(sx + lm2[na]["x"] * sw); ay = int(PY + 14 + lm2[na]["y"] * PH * 0.55)
                        bx = int(sx + lm2[nb]["x"] * sw); by = int(PY + 14 + lm2[nb]["y"] * PH * 0.55)
                        col = (255,220,80) if cur else (100,120,140)
                        pygame.draw.line(screen, col, (ax, ay), (bx, by), 2 if cur else 1)
            nby = PY + PH + 2
            bp = pygame.Rect(lx+int(W*0.35), nby, int(W*0.05), int(H*0.03))
            bn = pygame.Rect(lx+int(W*0.55), nby, int(W*0.05), int(H*0.03))
            if preview_idx > 0:
                pc = (100,150,255) if bp.collidepoint(mx,my) else (60,60,80)
                pygame.draw.rect(screen, pc, bp, border_radius=4)
                screen.blit(font_xs.render("<", True, (255,255,255)), (bp.centerx-4, bp.centery-6))
            if preview_idx < total-1:
                pc = (100,150,255) if bn.collidepoint(mx,my) else (60,60,80)
                pygame.draw.rect(screen, pc, bn, border_radius=4)
                screen.blit(font_xs.render(">", True, (255,255,255)), (bn.centerx-4, bn.centery-6))
            if click and bp.collidepoint(mx,my) and preview_idx > 0: preview_idx -= 1
            if click and bn.collidepoint(mx,my) and preview_idx < total-1: preview_idx += 1
            # Continuous scroll while holding keys
            if scroll_timer > 0: scroll_timer -= 1
            if scroll_timer <= 0 and parsed_data and "poses" in parsed_data:
                tp = len(parsed_data["poses"])
                if keys[pygame.K_LEFT] and preview_idx > 0:
                    preview_idx -= 1; scroll_timer = 4
                elif keys[pygame.K_RIGHT] and preview_idx < tp - 1:
                    preview_idx += 1; scroll_timer = 4
            hint = font_xs.render("<- -> or click", True, (120,120,140))
            screen.blit(hint, (lx+int(W*0.65), nby+2))
        pygame.display.flip(); clock.tick(30)
