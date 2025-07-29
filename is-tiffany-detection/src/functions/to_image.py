from is_msgs.image_pb2 import Image
import numpy as np
import cv2

def to_image(image: np.ndarray, encode_format: str = ".jpeg", compression_level: float = 0.8,) -> Image:
        if encode_format == ".jpeg":
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()
        cimage = cv2.imencode(ext=encode_format, img=image, params=params)
        return Image(data=cimage[1].tobytes())