import glob
import os
from argparse import ArgumentParser
from typing import List, Tuple

from inference import frames, detect

OUTPUT_FRAMES_DIR = 'data/output_frames'
OUTPUT_FRAMES_SUFFIX = '_output_frames'


def detect_video(frames_path: str, save_visualized: bool = False) -> Tuple[List[List[detect.Detection]], str]:
    paths: List[str] = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))

    output_frames_path = os.path.join(
        OUTPUT_FRAMES_DIR,
        os.path.basename(frames_path).strip(frames.VIDEO_FRAMES_SUFFIX) + OUTPUT_FRAMES_SUFFIX
    )

    if save_visualized:
        if os.path.exists(output_frames_path):
            print('Output frames already exist at {}'.format(output_frames_path))
            save_visualized = False
        else:
            os.mkdir(output_frames_path)

    frame_detections: List[List[detect.Detection]] = []

    print()
    for p in paths:
        print(p)
        if save_visualized:
            detections, viz_img = detect.detect_objects(p, print_detections=True, visualize=True)

            save_path = os.path.join(output_frames_path, os.path.basename(p))
            viz_img.save(save_path)
        else:
            detections, _ = detect.detect_objects(p, print_detections=True, visualize=False)

        frame_detections.append(detections)
        print('\n' * 3)

    return frame_detections, output_frames_path


def process_video(video_path: str, save_visualized: bool = False):
    frames_path = frames.frames_from_video(video_path)

    output_frames_path = detect_video(frames_path, save_visualized)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', '-v', required=True, help='Path to the video.')

    args = parser.parse_args()

    process_video(args.video, True)
