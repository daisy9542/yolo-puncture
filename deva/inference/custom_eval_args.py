from argparse import ArgumentParser

def add_custom_eval_args(parser: ArgumentParser):
    parser.add_argument("--video_name", type=str, required=True, help="Save folder of the video")
    parser.add_argument("--enable_reinforce", action='store_true', help="Enable memory reinforce")
    parser.add_argument("--forward_clip_frames", type=int, default=0, help="Number of frames to forward in-clip Consensus")