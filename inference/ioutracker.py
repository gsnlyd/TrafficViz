import json
from argparse import ArgumentParser
from typing import List, Tuple, NamedTuple
from inference import detect


class Tracklet:
    def __init__(self, frame: int,
                 score: float,
                 boxes: List[Tuple[float, float, float, float]]):
        self.boxes = boxes
        self.score = score
        self.frame = frame

    def __repr__(self):
        return ("Tracklet(frame=" + repr(self.frame) +
                ", score=" + repr(self.score) +
                ", boxes=" + repr(self.boxes) + ")")


def iou_calc(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    (box1_x1, box1_y1, box1_x2, box1_y2) = map(float, box1)
    (box2_x1, box2_y1, box2_x2, box2_y2) = map(float, box2)

    # Get the minimum intersection of these two boxes, with a box of negative size if they don't intersect
    intersection_x1 = max(box1_x1, box2_x1)
    intersection_y1 = max(box1_y1, box2_y1)
    intersection_x2 = min(box1_x2, box2_x2)
    intersection_y2 = min(box1_y2, box2_y2)

    # If no overlap, then IOU is 0
    if intersection_x2 - intersection_x1 <= 0 or intersection_y2 - intersection_y1 <= 0:
        return 0

    union_x1 = min(box1_x1, box2_x1)
    union_y1 = min(box1_y1, box2_y1)
    union_x2 = max(box1_x2, box2_x2)
    union_y2 = max(box1_y2, box2_y2)

    union_size = (union_x2 - union_x1) * (union_y2 - union_y1)
    intersection_size = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

    return float(intersection_size) / union_size


def tracklets_from_detections(detected_by_frame: List[List[detect.Detection]],
                              starting_frame: int,
                              score_thresh_low: float,
                              score_thresh_high: float,
                              iou_thresh: float,
                              required_frames_per_track: int) -> List[Tracklet]:
    tracks_found = []
    tracks_processing = []

    for frame, detections in enumerate(detected_by_frame, starting_frame):
        # Cull out detections below our threshold
        detection_set = [detection for detection in detections if detection.score >= score_thresh_low]

        tracks_altered = []

        for processing in tracks_processing:
            updated = False

            if detection_set:
                best = None
                best_iou = 0
                for detection in detection_set:
                    score = iou_calc(processing.boxes[-1], detection.box)
                    if best is None or score > best_iou:
                        best_iou = score
                        best = detection
                if best is not None and best_iou >= iou_thresh:
                    processing.boxes.append(best.box)
                    processing.score = max(processing.score, best.score)

                    tracks_altered.append(processing)
                    updated = True

                    # If we've used a detection, we don't associate it with other tracks
                    del detection_set[detection_set.index(best)]

            if not updated and (processing.score >= score_thresh_high and
                                len(processing.boxes) >= required_frames_per_track):
                tracks_found.append(processing)

        # Spawn new tracklets for each unmapped box
        new_tracks = [Tracklet(frame=frame, score=it.score, boxes=[it.box]) for it in detection_set]

        tracks_processing = tracks_altered + new_tracks

    # If we have any tracks left over, see if they fit
    for processing in tracks_processing:
        if processing.score >= score_thresh_high and len(processing.boxes) >= required_frames_per_track:
            tracks_found.append(processing)

    return sorted(tracks_found, key=lambda t: t.frame)


class PositionTracklet(NamedTuple):
    frame: int
    score: float
    positions: List[Tuple[int, int]]


def box_to_position(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (
        (box[0] + box[2]) // 2,
        (box[1] + box[3]) // 2,
    )


def get_position_tracklets(box_tracks: List[Tracklet]) -> List[PositionTracklet]:
    return [PositionTracklet(
        frame=track.frame,
        score=track.score,
        positions=[box_to_position(box) for box in track.boxes]
    ) for track in box_tracks]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--detections', '-d', type=str, required=True, help='Detection JSON on which to act.')
    parser.add_argument('--thresh-low', '-t', type=float, required=True,
                        help='Minimum score threshold for detections.')
    parser.add_argument('--thresh-high', '-T', type=float, required=True,
                        help='Maximum score threshold for detections.')
    parser.add_argument('--iou-thresh', '-o', type=float, required=True,
                        help='Minimum Intersection-Over-Union threshold for a track.')
    parser.add_argument('--frames-per-track', '-f', type=int, default=60,
                        help='Minimum contiguous frames required to recognize a track.')
    parser.add_argument('--save-to', '-s', type=str, default=None,
                        help='Filepath to write tracklets to.')

    args = parser.parse_args()

    detected_data = json.load(open(args.detections))

    frame_start = detected_data["start"]
    detections_found = [[detect.Detection(box=tuple(d['box']),
                                          label=d['label'],
                                          label_str=d['label_str'],
                                          score=d['score'])
                         for d in frame_data]
                        for frame_data in detected_data["detections"]]

    tracklets = tracklets_from_detections(
        detected_by_frame=detections_found,
        starting_frame=frame_start,
        score_thresh_low=args.thresh_low,
        score_thresh_high=args.thresh_high,
        iou_thresh=args.iou_thresh,
        required_frames_per_track=args.frames_per_track
    )

    print(tracklets)

    if args.save_to is not None:
        tracklets_json = {
            "tracklets": [{
                "frame": it.frame,
                "score": it.score,
                "boxes": it.boxes
            } for it in tracklets]
        }
        json.dump(tracklets_json, open(args.save_to))
