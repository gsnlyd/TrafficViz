import glob
import math
import os
import time
from argparse import ArgumentParser
from typing import List, Tuple, Optional

from torch import multiprocessing

from inference import frames, detect

OUTPUTS_DIR = 'data/outputs'
VIDEO_OUTPUTS_SUFFIX = '_outputs'
VISUALIZED_DETECTIONS_SUBDIR = 'visualized_frames/detections'

DEFAULT_NUM_THREADS = 3


def __detect_frames(frame_detections: List[List[detect.Detection]], start_index: int,
                    paths: List[str], visualized_detections_path: str, save_visualized: bool = False):
    for i, p in enumerate(paths):
        start_time = time.time()

        if save_visualized:
            detections, viz_img = detect.detect_objects(p, visualize=True)

            save_path = os.path.join(visualized_detections_path, os.path.basename(p))
            viz_img.save(save_path)
        else:
            detections, _ = detect.detect_objects(p, visualize=False)

        print('Detections completed for {} in {:.2f}s'.format(p, time.time() - start_time))

        path_i = start_index + i
        frame_detections[path_i] = detections


def detect_video(frames_path: str, video_outputs_dir: str,
                 num_threads: int = DEFAULT_NUM_THREADS,
                 save_visualized: bool = False, save_detections: bool = False,
                 start: int = 0, end: int = None) -> List[List[detect.Detection]]:
    assert start >= 0
    assert end is None or end >= 0

    paths: List[str] = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    if end is None:
        paths = paths[start:]
    else:
        paths = paths[start:end]

    visualized_detections_path = os.path.join(
        video_outputs_dir,
        VISUALIZED_DETECTIONS_SUBDIR
    )
    os.makedirs(visualized_detections_path)

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
            visualized_detections_path,
            save_visualized
        ))
        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    assert None not in frame_detections
    frame_detections: List[List[detect.Detection]]
    return frame_detections


def get_outputs_dir(video_name: str) -> str:
    def outputs_dir_name(num: int) -> str:
        return os.path.join(
            OUTPUTS_DIR,
            video_name + VIDEO_OUTPUTS_SUFFIX + '_' + str(num)
        )

    i = 1
    cur_name = outputs_dir_name(i)
    while os.path.exists(cur_name):
        i += 1
        cur_name = outputs_dir_name(i)

    return cur_name


def process_video(video_path: str, num_threads: int = DEFAULT_NUM_THREADS,
                  save_visualized: bool = False, save_detections: bool = False,
                  start: int = 0, end: int = None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_outputs_dir = get_outputs_dir(video_name)
    os.makedirs(video_outputs_dir)

    frames_path = frames.frames_from_video(video_path)

    detect_video(
        frames_path=frames_path,
        video_outputs_dir=video_outputs_dir,
        num_threads=num_threads,
        save_visualized=save_visualized,
        save_detections=save_detections,
        start=start,
        end=end
    )


if __name__ == '__main__':
    assert os.path.exists('data')

    parser = ArgumentParser()
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to the video.')
    parser.add_argument('--num-threads', '--threads', type=int, default=DEFAULT_NUM_THREADS)

    parser.add_argument('--save-visualizations', '-sv', action='store_true',
                        help='Save frames with detections visualized.')
    parser.add_argument('--save-detections', '-sd', action='store_true',
                        help='Save raw detections in JSON format.')

    parser.add_argument('--start', '-s', type=int, default=0, help='Frame to start on.')
    parser.add_argument('--end', '-e', type=int, default=None, help='Frame to end on.')

    args = parser.parse_args()
    print(args)

    process_video(
        video_path=args.video,
        num_threads=args.num_threads,
        save_visualized=args.save_visualizations,
        save_detections=args.save_detections,
        start=args.start - 1,
        end=None if args.end is None else args.end - 1
    )
