import glob
import math
import os
import time
from argparse import ArgumentParser
from typing import List, Tuple, Optional

from torch import multiprocessing

from inference import frames, detect

OUTPUT_FRAMES_DIR = 'data/output_frames'
OUTPUT_FRAMES_SUFFIX = '_output_frames'

DEFAULT_NUM_THREADS = 3


def __detect_frames(frame_detections: List[List[detect.Detection]], start_index: int,
                    paths: List[str], output_frames_path: str, save_visualized: bool = False):
    for i, p in enumerate(paths):
        start_time = time.time()

        if save_visualized:
            detections, viz_img = detect.detect_objects(p, visualize=True)

            save_path = os.path.join(output_frames_path, os.path.basename(p))
            viz_img.save(save_path)
        else:
            detections, _ = detect.detect_objects(p, visualize=False)

        print('Detections completed for {} in {:.2f}s'.format(p, time.time() - start_time))

        path_i = start_index + i
        frame_detections[path_i] = detections


def detect_video(frames_path: str, save_visualized: bool = False, num_threads: int = DEFAULT_NUM_THREADS) -> \
        Tuple[List[List[detect.Detection]], str]:
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

    print()

    manager = multiprocessing.Manager()
    frame_detections: List[Optional[List[detect.Detection]]] = manager.list([None] * len(paths))

    paths_per_thread = math.ceil(len(paths) / num_threads)

    path_slices = []
    start_indices = []

    for i in range(0, len(paths), paths_per_thread):
        path_slices.append(paths[i:i + paths_per_thread])
        start_indices.append(i)

    processes = []
    for s, s_i in zip(path_slices, start_indices):
        p = multiprocessing.Process(target=__detect_frames, args=(
            frame_detections,
            s_i,
            s,
            output_frames_path,
            save_visualized
        ))
        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    return frame_detections


def process_video(video_path: str, save_visualized: bool = False, num_threads: int = DEFAULT_NUM_THREADS):
    frames_path = frames.frames_from_video(video_path)

    output_frames_path = detect_video(frames_path, save_visualized, num_threads)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to the video.')
    parser.add_argument('--num-threads', '--threads', type=int, default=DEFAULT_NUM_THREADS)

    args = parser.parse_args()

    process_video(args.video, save_visualized=True, num_threads=args.num_threads)
