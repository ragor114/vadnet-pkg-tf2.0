import numpy as np
import os
import librosa as lr


def audio_from_file(path: str, sr: int = 48000, offset: float = 0.0, duration: float = None) -> np.array:
    try:
        audio, _ = lr.load(
            path,
            sr=sr,
            mono=True,
            offset=offset,
            duration=duration,
            dtype=np.float32,
            res_type="kaiser_fast",
        )
        audio.shape = (-1, 1)
        return audio
    except ValueError as ex:
        print("value error {}\n{}".format(path, ex))
        return []
    except Exception as ex:
        print("could not read {}\n{}".format(path, ex))
        return None


def audio_to_frames(x: np.array, n_frame: int, n_step: int = None) -> np.array:
    if n_step is None:
        n_step = n_frame
    if len(x.shape) == 1:
        x.shape = (-1, 1)
    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step
    n_keep = n_frames * n_step + n_overlap
    strides = list(x.strides)
    strides[0] = strides[1] * n_step
    return np.lib.stride_tricks.as_strided(
        x[0:n_keep, :], (n_frames, n_frame), strides
    )
