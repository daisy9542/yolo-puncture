from argparse import ArgumentParser

def add_custom_eval_args(parser: ArgumentParser):
    parser.add_argument("--enable_reinforce", action='store_true', default=False, help="Enable memory reinforce")
    parser.add_argument("--forward_clip_frames", type=int, default=0, help="Number of frames to forward in-clip Consensus")
    parser.add_argument("--keyframe_selection", type=str, default="first", help="Keyframe selection method, first | last | middle | score")