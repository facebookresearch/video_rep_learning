import decord
from decord import VideoReader
from decord import cpu, gpu

def decord_load(file_path, start, end, device_id=0):
    decord.bridge.set_bridge('torch')
    # vr = VideoReader(file_path, ctx=gpu(0))
    vr = VideoReader(file_path, ctx=cpu(0))
    frames = vr.get_batch(range(start, end))
    return frames