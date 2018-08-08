import numpy as np
import cv2
from functools import reduce
import os

class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def none_or_str(value):
    if value == 'None':
        return None
    return value


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def str_as_bool(value):
    if value in ['True', 'true', 't', '1']:
        return True
    elif value in ['False', 'false', 'f', '0']:
        return False
    else:
        raise ValueError("Argument must either be the string, \'True\' or \'False\'")

def convert_scientific_str_to_int(value):
    return int(float(value))

def pretty_print_dict(dict, prefix):
    return reduce(lambda acc, key: '{}\n{}: {}'.format(acc, key, dict[key]), dict.keys(), prefix)


def debug_imshow_image_with_caption(window_label, frame, caption):
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (0, 10)
    font_scale = 0.3
    font_color = (0, 0, 0)
    line_type = 1

    frame = np.copy(frame)

    cv2.putText(frame, str(caption),
                location,
                font,
                font_scale,
                font_color,
                line_type)

    cv2.imshow(window_label, frame[:, :, ::-1])


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def convertToOneHot(vector, num_classes):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    # assert isinstance(vector, np.ndarray)
    # assert len(vector) > 0
    # print("onehot input: {}".format(vector))

    vector = np.reshape(np.asarray(vector, dtype=np.int32), newshape=[-1])

    assert len(vector.shape) == 1

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    result = np.squeeze(result)
    # print("onehot result: {}".format(result))

    return result.astype(np.float32)