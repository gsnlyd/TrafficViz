import os
import subprocess
from argparse import ArgumentParser

VIDEOS_DIR = 'data/videos'
FRAMES_DIR = 'data/video_frames'
VIDEO_FRAMES_SUFFIX = '_frames'
OUTPUT_PATH_SUFFIX = '_frame_%05d.jpg'

JPG_QUALITY = 2
DEFAULT_FRAMERATE = 2


def frames_from_video(video_path: str, framerate: int = DEFAULT_FRAMERATE) -> str:
    assert os.path.exists(video_path), 'Video does not exist at path {}'.format(video_path)
    assert 0 < framerate, 'Invalid framerate "{}" - must be positive'.format(framerate)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_frames_path = os.path.join(FRAMES_DIR, video_name + VIDEO_FRAMES_SUFFIX)

    if os.path.exists(video_frames_path):
        print('Frames already exist for video {} at path {}'.format(video_name, video_frames_path))
        return video_frames_path

    os.mkdir(video_frames_path)
    output_path = os.path.join(video_frames_path, video_name + OUTPUT_PATH_SUFFIX)

    ffmpeg_command = 'ffmpeg -i {} -qscale:v {} -r {} {}'.format(
        video_path,
        JPG_QUALITY,
        framerate,
        output_path
    )

    print('Running command: {}'.format(ffmpeg_command))
    subprocess.run(ffmpeg_command, shell=True)  # TODO: Don't use shell mode

    num_frames_created = len(os.listdir(video_frames_path))
    assert num_frames_created > 0

    print('Created {} frames for video {} in directory {}'.format(num_frames_created, video_path, video_frames_path))
    return video_frames_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', '-v', type=str, required=True, help='Path to video.')
    parser.add_argument('--framerate', '-f', type=int, default=DEFAULT_FRAMERATE,
                        help='Framerate at which to output video frames.')

    args = parser.parse_args()
    print(args)

    frames_from_video(args.video_path)
