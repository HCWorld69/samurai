import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import sys
import time
import gc

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def load_video_frames(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames were loaded from the video.")
    return frames


def select_bbox(frame):
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    x, y, w, h = [int(v) for v in bbox]
    return (x, y, x + w, y + h)


def compute_bbox_from_mask(mask):
    mask = mask[0].cpu().numpy() > 0.0
    non_zero = np.argwhere(mask)
    if len(non_zero) == 0:
        return [0, 0, 0, 0]
    y_min, x_min = non_zero.min(axis=0).tolist()
    y_max, x_max = non_zero.max(axis=0).tolist()
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {'GPU' if device.startswith('cuda') else 'CPU'}")

    try:
        frames = load_video_frames(args.video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    first_frame = frames[0]
    if args.box is None:
        bbox = select_bbox(first_frame)
    else:
        x1, y1, x2, y2 = map(int, args.box.split(','))
        bbox = (x1, y1, x2, y2)
    print(f"Selected bbox: {bbox}")

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device)

    with torch.inference_mode():
        state = predictor.init_state(frames, offload_video_to_cpu=True)
        start = time.time()
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        print(f"Frame 0 processed in {time.time() - start:.2f}s")
        bbox_log = {0: compute_bbox_from_mask(masks[0])}

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            if frame_idx >= 5:
                break
            start = time.time()
            for obj_id, mask in zip(obj_ids, masks):
                bbox_log[frame_idx] = compute_bbox_from_mask(mask)
            print(f"Frame {frame_idx} processed in {time.time() - start:.2f}s, bbox: {bbox_log[frame_idx]}")

    del predictor, state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Model checkpoint path")
    parser.add_argument("--box", help="Optional initial bounding box x1,y1,x2,y2")
    args = parser.parse_args()
    main(args)
