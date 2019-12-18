import glob
import json
import math
import os
import time
from argparse import ArgumentParser
from typing import List, Tuple, Optional

from PIL import Image
from torch import multiprocessing

from inference import frames, detect
from inference.detect import Detector

OUTPUTS_DIR = 'data/outputs'
VIDEO_OUTPUTS_SUFFIX = '_outputs'
VISUALIZED_DETECTIONS_SUBDIR = 'visualized_frames/detections'
DETECTIONS_FILE_SUFFIX = '_detections.json'

DEFAULT_NUM_THREADS = 3


def save_detections_file(path: str, start: int, end: int, detections: List[List[detect.Detection]]):
    detections_json = {
        'start': start,
        'end': end,
        'detections': [[d._asdict() for d in l] for l in detections]
    }

    with open(path, 'w') as detections_file:
        json.dump(detections_json, detections_file, indent=2)


def load_detections_file(path: str) -> Tuple[int, int, List[List[detect.Detection]]]:
    assert os.path.exists(path)

    with open(path) as detections_file:
        detections_json = json.load(detections_file)
        start: int = detections_json['start']
        end: int = detections_json['end']

        detections: List[List[detect.Detection]] = []

        for l in detections_json['detections']:
            d_l = []
            for d in l:
                d_l.append(detect.Detection(
                    box=tuple(d['box']),
                    label=d['label'],
                    label_str=d['label_str'],
                    score=d['score']
                ))
            detections.append(d_l)

    return start, end, detections


def detect_video(video_name: str, frames_path: str, video_outputs_dir: str, detector: Detector,
                 num_threads: int = DEFAULT_NUM_THREADS,
                 start: int = 0, end: int = None) -> List[List[detect.Detection]]:
    assert start >= 0
    assert end is None or end >= 0

    paths: List[str] = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    if end is None:
        end = len(paths)
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

    def detect_subset(subset_paths: List[str], start_index: int):
        for subset_path_i, path in enumerate(subset_paths):
            s_time = time.time()
            im: Image.Image = Image.open(path)

            det = detector.detect_objects(im)
            detect.visualize(im, det)
            # Save detections from this thread into shared list
            frame_detections[start_index + subset_path_i] = det
            print('{} detections for frame {} ({:.2f}s)'.format(len(det), path, time.time() - s_time))

            # Save visualized frames
            save_path = os.path.join(visualized_detections_path, os.path.basename(path))
            im.save(save_path)

    processes = []
    # Start threads
    for s, s_i in zip(path_slices, start_indices):
        p = multiprocessing.Process(target=detect_subset, args=(s, s_i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    assert None not in frame_detections  # Sanity check: ensure all threads did their job
    frame_detections: List[List[detect.Detection]]

    # Save detections
    detections_file_path = os.path.join(video_outputs_dir, video_name + DETECTIONS_FILE_SUFFIX)
    save_detections_file(detections_file_path, start, end, frame_detections)
    print('\nSaved detections at {}'.format(detections_file_path))

    return frame_detections


def get_new_outputs_dir(video_name: str) -> str:
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


def process_video(video_path: str,
                  framerate: int = frames.DEFAULT_FRAMERATE,
                  finetuned_path: str = None,
                  num_threads: int = DEFAULT_NUM_THREADS,
                  start: int = 0, end: int = None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_outputs_dir = get_new_outputs_dir(video_name)
    os.makedirs(video_outputs_dir)

    frames_path = frames.frames_from_video(video_path, framerate)

    detector = Detector(finetuned_path=finetuned_path)
    detections = detect_video(
        video_name=video_name,
        frames_path=frames_path,
        video_outputs_dir=video_outputs_dir,
        detector=detector,
        num_threads=num_threads,
        start=start,
        end=end
    )


if __name__ == '__main__':
    assert os.path.exists('data')

    parser = ArgumentParser()
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to the video.')
    parser.add_argument('--framerate', '-r', type=int, default=frames.DEFAULT_FRAMERATE, help='Video framerate.')
    parser.add_argument('--num-threads', '--threads', type=int, default=DEFAULT_NUM_THREADS)

    parser.add_argument('--finetuned-path', '-f', type=str, default=None,
                        help='Path to load fine-tuned model weights.')

    parser.add_argument('--start', '-s', type=int, default=0, help='Frame to start on.')
    parser.add_argument('--end', '-e', type=int, default=None, help='Frame to end on.')

    args = parser.parse_args()
    print(args)

    process_video(
        video_path=args.video,
        framerate=args.framerate,
        finetuned_path=args.finetuned_path,
        num_threads=args.num_threads,
        start=args.start - 1,
        end=None if args.end is None else args.end - 1
    )
