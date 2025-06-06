from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_msgs.image_pb2 import Image, ObjectAnnotations
from opencensus.trace.blank_span import BlankSpan
from typing import Dict, Tuple
from is_wire.core import Tracer
from .to_np import to_np
from classes import StreamChannel
import numpy as np
import re

def get_images_from_camera(channel_camera: StreamChannel, channel_detection: StreamChannel, exporter: ZipkinExporter, conf: float, camera_id: int) -> Tuple[np.ndarray, Tracer, BlankSpan, int, np.ndarray]:
    '''
    Obtém a imagem recortada (ROI) da detecção da câmera.

    Parâmetros:
        channel (Dict[int, StreamChannel]): Dicionário de canais de stream indexados por ID da câmera.
        exporter (Dict[int, ZipkinExporter]): Dicionário de exportadores Zipkin por ID da câmera.
        conf (float): Valor mínimo de score para aceitar o bounding box.
        camera_id (int): Valor do ID da câmera.
    Retorna:
        Tuple contendo:
            - crop (np.ndarray): Imagem recortada (ROI) com base na primeira detecção.
            - tracer (Tracer): Objeto Tracer para monitoramento distribuído.
            - span (BlankSpan): Span de trace ativo para a operação.
            - camera_id (int): ID da câmera que enviou a imagem.
            - roi_offset (np.ndarray): Coordenadas (x1, y1) do canto superior esquerdo do ROI na imagem original.
            - original_image (np.ndarray): Imagem completa da câmera
    '''

    while True:
        obj_det = channel_detection.consume_last()

        if not isinstance(obj_det, bool):
            det = obj_det.unpack(ObjectAnnotations)
            score: float = det.objects[0].score
            if score < conf:
                continue

            tracer: Tracer = Tracer(
                exporter = exporter,
                span_context = obj_det.extract_tracing()
            )
            span: BlankSpan = tracer.start_span(name = "tiffany_keypoints_detection")

            with tracer.span(name = "get_and_unpack_image_from_camera"):
                while True:
                    image = channel_camera.consume_last()

                    if not isinstance(image, bool):
                        img = image.unpack(Image)
                        img_np = to_np(img)
                        original_image = img_np
                        box = det.objects[0].region.vertices
                        x1, y1 = int(box[0].x), int(box[0].y)
                        x2, y2 = int(box[1].x), int(box[1].y)

                        crop = img_np[y1:y2, x1:x2]
                        roi_offset = np.array([x1, y1])
                        return crop, tracer, span, roi_offset, original_image
