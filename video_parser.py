"""Video pose extractor: process a video file, extract pose landmarks per frame,
and save as a pose sequence for the follow-along mode."""

import json
import os
import cv2
import numpy as np
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


def extract_poses(video_path: str, output_path: str,
                  sample_every: int = 3, model_path: str = None):
    """Extract pose landmarks from video, save as JSON pose sequence.

    Args:
        video_path: path to input video file
        output_path: path to save JSON output
        sample_every: process every Nth frame (default 3 = ~10fps at 30fps)
        model_path: path to pose_landmarker.task model
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_v2.task")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps, sampling every {sample_every} frame(s)")

    # Init MediaPipe
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    pose_sequence = []
    frame_idx = 0
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # Compute timestamp from actual frame index BEFORE detection
        ts_now = int(frame_idx * 1000 / fps) if fps > 0 else frame_idx * 33
        result = landmarker.detect_for_video(mp_image, ts_now)

        if result.pose_landmarks:
            frame_data = {"frame": frame_idx, "time_ms": ts_now}
            landmarks_dict = {}
            for i, lm in enumerate(result.pose_landmarks[0]):
                if i < len(LM_NAMES):
                    landmarks_dict[LM_NAMES[i]] = {
                        "x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility
                    }
            frame_data["landmarks"] = landmarks_dict
            pose_sequence.append(frame_data)

        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames, {len(pose_sequence)} poses found")

    landmarker.close()
    cap.release()

    # Save
    output = {
        "source": video_path,
        "total_frames": total_frames,
        "sampled_frames": len(pose_sequence),
        "poses": pose_sequence,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Saved {len(pose_sequence)} poses to {output_path}")
    return output_path


def extract_key_poses(input_path: str, output_path: str,
                      min_displacement: float = 0.08, min_interval: int = 10):
    """Extract distinct key poses from a pose sequence.

    Args:
        input_path: JSON file from extract_poses
        output_path: path for filtered key poses
        min_displacement: minimum avg joint movement to count as new pose
        min_interval: minimum frames between key poses
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    poses = data["poses"]
    if not poses:
        print("No poses to filter")
        return output_path

    key_poses = [poses[0]]
    last_frame = poses[0]["frame"]

    for pose in poses[1:]:
        if pose["frame"] - last_frame < min_interval:
            continue

        # Compare with last key pose
        prev_lm = key_poses[-1]["landmarks"]
        curr_lm = pose["landmarks"]
        total_dist = 0.0
        count = 0
        key_joints = ["LEFT_WRIST","RIGHT_WRIST","LEFT_ELBOW","RIGHT_ELBOW",
                      "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
        for name in key_joints:
            if name in prev_lm and name in curr_lm:
                dx = prev_lm[name]["x"] - curr_lm[name]["x"]
                dy = prev_lm[name]["y"] - curr_lm[name]["y"]
                total_dist += abs(dx) + abs(dy)
                count += 1

        if count > 0 and total_dist / count > min_displacement:
            key_poses.append(pose)
            last_frame = pose["frame"]

    output = {
        "source": data["source"],
        "total_frames": data["total_frames"],
        "sampled_frames": len(poses),
        "key_poses": len(key_poses),
        "poses": key_poses,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Extracted {len(key_poses)} key poses from {len(poses)} frames -> {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_parser.py <video_path> [output_path]")
        sys.exit(1)
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else video + ".poses.json"
    extract_poses(video, out)
    key_out = out.replace(".json", ".key.json")
    extract_key_poses(out, key_out)
