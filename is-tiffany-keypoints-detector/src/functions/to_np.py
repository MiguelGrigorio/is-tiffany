import cv2
import numpy as np
from is_msgs.image_pb2 import Image
from typing import Union

def to_np(input_image: Union[np.ndarray, Image]) -> np.ndarray:
    """
    Converte uma imagem no formato `is_msgs.image_pb2.Image` ou `np.ndarray` para um `np.ndarray` padrão OpenCV.

    Parâmetros:
        input_image (Union[np.ndarray, Image]): Imagem no formato IS ou numpy.

    Retorna:
        np.ndarray: Imagem decodificada no formato BGR (OpenCV), ou um array vazio se o tipo não for suportado.
    """
    if isinstance(input_image, np.ndarray):
        return input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
        return output_image if output_image is not None else np.array([], dtype=np.uint8)
    else:
        return np.array([], dtype=np.uint8)
