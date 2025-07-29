from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_msgs.image_pb2 import Image
from opencensus.trace.blank_span import BlankSpan
from typing import Tuple
from is_wire.core import Tracer
from .to_np import to_np
from classes import StreamChannel
import numpy as np

def get_images_from_camera(channel_camera: StreamChannel, exporter: ZipkinExporter) -> Tuple[np.ndarray, Tracer, BlankSpan]:
    '''
    Obtém a imagem recortada (ROI) da detecção da câmera.

    Parâmetros:
        channel_camera (Dict[int, StreamChannel]): Dicionário de canais de stream indexados por ID da câmera.
        exporter (Dict[int, ZipkinExporter]): Dicionário de exportadores Zipkin por ID da câmera.
    Retorna:
        Tuple contendo:
            - img_np (np.ndarray): Imagem completa da câmera.
            - tracer (Tracer): Objeto Tracer para monitoramento distribuído.
            - span (BlankSpan): Span de trace ativo para a operação.
    '''

    while True:
        while True:
            image = channel_camera.consume(0.0)
            
            if not isinstance(image, bool):
                tracer: Tracer = Tracer(
                    exporter = exporter, 
                    span_context = image.extract_tracing()
                    )
                span: BlankSpan = tracer.start_span(name = "tiffany_detection")

                with tracer.span(name = "get_and_unpack_image_from_camera"):
                    img = image.unpack(Image)
                    img_np = to_np(img)
                    return img_np, tracer, span
