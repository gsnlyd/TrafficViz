import glob
import json
import math
import os
import random
import time
from argparse import ArgumentParser
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw
from torch import multiprocessing

from inference import frames, detect, ioutracker
from inference.detect import Detector
from inference.ioutracker import Tracklet, PositionTracklet

OUTPUTS_DIR = 'data/outputs'
VIDEO_OUTPUTS_SUFFIX = '_outputs'
VISUALIZED_DETECTIONS_SUBDIR = 'visualized_frames/detections'
VISUALIZED_TRACKS_SUBDIR = 'visualized_frames/tracks'
DETECTIONS_FILE_SUFFIX = '_detections.json'

TRACKS_VIDEO_SUFFIX = '_tracks.mp4'

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


def detect_video(video_name: str, paths: List[str], video_outputs_dir: str, detector: Detector,
                 start: int, end: int,
                 num_threads: int = DEFAULT_NUM_THREADS) -> List[List[detect.Detection]]:
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


def visualize_tracks(box_tracks: List[Tracklet], position_tracks: List[PositionTracklet], paths: List[str],
                     start: int, end: int,
                     save_dir: str):
    def random_color() -> Tuple:
        return tuple([random.randrange(0, 256) for i in range(3)])

    track_colors = [random_color() for i in range(len(box_tracks))]

    for path_i, path in enumerate(paths):
        img = Image.open(path)
        draw = ImageDraw.Draw(img)

        frame = start + path_i
        for track_i, (b, p) in enumerate(zip(box_tracks, position_tracks)):
            if frame < b.frame:
                break

            track_end = b.frame + len(b.boxes)
            if frame >= track_end:
                continue

            cur_track_frame = frame - b.frame
            cur_box = [int(v) for v in b.boxes[cur_track_frame]]
            positions = p.positions[:cur_track_frame + 1]

            track_c = track_colors[track_i]
            draw.rectangle(cur_box, outline=track_c, width=3)
            for pos in positions:
                xy = (
                    int(pos[0] - 2),
                    int(pos[1] - 2),
                    int(pos[0] + 2),
                    int(pos[1] + 2)
                )
                draw.ellipse(xy, fill=track_c)
        save_path = os.path.join(save_dir, os.path.basename(path))
        img.save(save_path)


def outputs_dir_name(video_name: str, num: int) -> str:
    return os.path.join(
        OUTPUTS_DIR,
        video_name + VIDEO_OUTPUTS_SUFFIX + '_' + str(num)
    )


def get_new_outputs_dir(video_name: str) -> str:
    i = 1
    cur_name = outputs_dir_name(video_name, i)
    while os.path.exists(cur_name):
        i += 1
        cur_name = outputs_dir_name(video_name, i)

    return cur_name


def process_video(video_path: str,
                  framerate: int = frames.DEFAULT_FRAMERATE,
                  finetuned_path: str = None,
                  use_detections: int = None,
                  num_threads: int = DEFAULT_NUM_THREADS,
                  start: int = 0, end: int = None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_outputs_dir = get_new_outputs_dir(video_name)
    os.makedirs(video_outputs_dir)

    frames_path = frames.frames_from_video(video_path, framerate)

    paths: List[str] = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    if end is None:
        end = len(paths)
    paths = paths[start:end]

    if use_detections is None:
        detector = Detector(finetuned_path=finetuned_path)
        detections = detect_video(
            video_name=video_name,
            paths=paths,
            video_outputs_dir=video_outputs_dir,
            detector=detector,
            start=start,
            end=end,
            num_threads=num_threads
        )
    else:
        previous_outputs_dir = outputs_dir_name(video_name, use_detections)
        assert os.path.exists(previous_outputs_dir)
        print('Using detections from previous output: {}'.format(previous_outputs_dir))

        detections_file_path = os.path.join(previous_outputs_dir, video_name + DETECTIONS_FILE_SUFFIX)
        start, end, detections = load_detections_file(detections_file_path)

    box_tracks = ioutracker.tracklets_from_detections(detections,
                                                      starting_frame=start,
                                                      score_thresh_low=0,
                                                      score_thresh_high=0,
                                                      iou_thresh=0.1,
                                                      required_frames_per_track=2)
    position_tracks = ioutracker.get_position_tracklets(box_tracks)

    print('Found {} tracks from {} frames with {} detections'.format(
        len(box_tracks),
        len(paths),
        sum([len(d) for d in detections])
    ))

    track_viz_dir = os.path.join(video_outputs_dir, VISUALIZED_TRACKS_SUBDIR)
    os.makedirs(track_viz_dir)
    visualize_tracks(
        box_tracks=box_tracks,
        position_tracks=position_tracks,
        paths=paths,
        start=start,
        end=end,
        save_dir=track_viz_dir
    )

    track_frames_name = os.path.join(
        video_outputs_dir,
        VISUALIZED_TRACKS_SUBDIR,
        video_name + frames.OUTPUT_PATH_SUFFIX
    )
    track_video_save_path = os.path.join(video_outputs_dir, video_name + TRACKS_VIDEO_SUFFIX)

    frames.video_from_frames(track_frames_name,
                             start=start,
                             framerate=framerate,
                             save_path=track_video_save_path)
    print('Finished processing {}'.format(video_path))


if __name__ == '__main__':
    assert os.path.exists('data')

    parser = ArgumentParser()
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to the video.')
    parser.add_argument('--framerate', '-r', type=int, default=frames.DEFAULT_FRAMERATE, help='Video framerate.')
    parser.add_argument('--num-threads', '--threads', type=int, default=DEFAULT_NUM_THREADS)

    parser.add_argument('--finetuned-path', '-f', type=str, default=None,
                        help='Path to load fine-tuned model weights.')

    parser.add_argument('--detections', '-d', type=int, default=None,
                        help='Previous output number from which to load detections.')

    parser.add_argument('--start', '-s', type=int, default=1, help='Frame to start on.')
    parser.add_argument('--end', '-e', type=int, default=None, help='Frame to end on.')

    args = parser.parse_args()
    print(args)

    process_video(
        video_path=args.video,
        framerate=args.framerate,
        finetuned_path=args.finetuned_path,
        use_detections=args.detections,
        num_threads=args.num_threads,
        start=args.start - 1,
        end=None if args.end is None else args.end - 1
    )
