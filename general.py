import numpy as np


def ascii_to_str(array):
    """
    Get a string from an array of ascii integers

    Parameters
    ----------
    array: np.ndarray
        array of integers representing ascii chars
    Returns
    -------
    str
        string represented by the array
    """
    if not hasattr(array, '__iter__'):
        return array
    return ''.join(map(chr, filter(None, array)))


def find_frame_index_from_timestamp(timestamp, frame_timestamps):
    """
    Find the index of the microscope frame, that precedes the provided timestamp.

    Parameters
    ----------
    timestamp: float
        Timestamp of the event you want to find the corresponding frame to, e.g. stimulation start.
    frame_timestamps: list, np.ndarray
        Timestamps of all the recorded microscope frames.

    Returns
    -------
    int
        Index of the frame preceding the given timestamp or None if the timestamp falls out of the range of the frame
        timestamps.
    """
    if timestamp < frame_timestamps[0]:
        return None
    elif timestamp > frame_timestamps[-1]:
        return None

    for i in range(len(frame_timestamps) - 1):
        if frame_timestamps[i] <= timestamp < frame_timestamps[i + 1]:
            return i

