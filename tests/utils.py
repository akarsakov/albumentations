from io import StringIO

import cv2
import numpy as np


def convert_2d_to_3d(arrays, num_channels=3):
    # Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, num_channels)
    # by repeating the existing values along the new axis.
    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target_format(arrays, target):
    if target == "mask":
        return arrays[0] if len(arrays) == 1 else arrays
    if target == "image":
        return convert_2d_to_3d(arrays, num_channels=3)
    if target == "image_4_channels":
        return convert_2d_to_3d(arrays, num_channels=4)

    raise ValueError("Unknown target {}".format(target))


class InMemoryFile(StringIO):
    def __init__(self, value, save_value, file):
        super().__init__(value)
        self.save_value = save_value
        self.file = file

    def close(self):
        self.save_value(self.getvalue(), self.file)
        super().close()


class OpenMock:
    """
    Mocks the `open` built-in function. A call to the instance of OpenMock returns an in-memory file which is
    readable and writable. The actual in-memory file implementation should call the passed `save_value` method
    to save the file content in the cache when the file is being closed to preserve the file content.
    """

    def __init__(self):
        self.values = {}

    def __call__(self, file, *args, **kwargs):
        value = self.values.get(file)
        return InMemoryFile(value, self.save_value, file)

    def save_value(self, value, file):
        self.values[file] = value


class ImreadMock:
    """
    Mocks the `cv2.imread`. A call to the instance of ImreadMock returns a predefined image
    or reads actual image from file
    """

    IMG_100_8UC1 = "template_100_u8c1.png"
    IMG_224_8UC1 = "template_224_u8c1.png"
    IMG_512_8UC1 = "template_512_u8c1.png"
    IMG_512_8UC3 = "template_512_u8c3.png"

    def __init__(self):
        self.images = {
            self.IMG_100_8UC1: np.random.randint(0, 256, size=(100, 100), dtype=np.uint8),
            self.IMG_224_8UC1: np.random.randint(0, 256, size=(224, 224), dtype=np.uint8),
            self.IMG_512_8UC1: np.random.randint(0, 256, size=(512, 512), dtype=np.uint8),
            self.IMG_512_8UC3: np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8),
        }

    def __call__(self, file, *args, **kwargs):
        if file in self.images:
            return self.images[file]

        return cv2.imread(file, *args, **kwargs)
