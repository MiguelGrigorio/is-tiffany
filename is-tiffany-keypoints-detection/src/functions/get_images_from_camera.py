from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_msgs.image_pb2 import Image, ObjectAnnotations
from opencensus.trace.blank_span import BlankSpan
from typing import Tuple
from is_wire.core import Tracer
from .to_np import to_np
from classes import StreamChannel
import numpy as np

def get_images_from_camera(channel_camera: StreamChannel, channel_detection: StreamChannel, exporter: ZipkinExporter, conf: float) -> Tuple[np.ndarray, Tracer, BlankSpan]:
    '''
    Obtém a imagem recortada (ROI) da detecção da câmera.

    Parâmetros:
        channel_camera (StreamChannel): Canal de stream da câmera.
        channel_detection (StreamChannel): Canal de stream para detecções.
        exporter (ZipkinExporter): Exportador Zipkin para rastreamento.
        conf (float): Confiança mínima para considerar uma detecção válida.

    Retorna:
        Tuple contendo:
            - crop (np.ndarray): Imagem recortada (ROI) da câmera.
            - tracer (Tracer): Objeto Tracer para monitoramento distribuído.
            - span (BlankSpan): Span de trace ativo para a operação.
            - roi_offset (np.ndarray): Coordenadas (x1, y1) do canto superior esquerdo do ROI na imagem original.
    '''
    while True:
        obj_det = channel_detection.consume_last()

        if not isinstance(obj_det, bool):
            det = obj_det.unpack(ObjectAnnotations)
            score: float = det.objects[0].score
            if score < conf:
                continue

            tracer: Tracer = Tracer(
                exporter=exporter,
                span_context=obj_det.extract_tracing()
            )
            span: BlankSpan = tracer.start_span(name="tiffany_keypoints_detection")

            with tracer.span(name="get_and_unpack_image_from_camera"):
                while True:
                    image = channel_camera.consume_last()

                    if not isinstance(image, bool):
                        img = image.unpack(Image)
                        img_np = to_np(img)
                        box = det.objects[0].region.vertices
                        x1, y1 = int(box[0].x), int(box[0].y)
                        x2, y2 = int(box[1].x), int(box[1].y)
                        
                        crop = img_np[y1:y2, x1:x2]
                        roi_offset = np.array([x1, y1])
                        return crop, tracer, span, roi_offset
