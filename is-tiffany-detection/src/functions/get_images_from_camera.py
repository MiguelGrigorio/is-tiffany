from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.core import Tracer, Message
from opencensus.trace.span import Span
from is_msgs.image_pb2 import Image
from classes import StreamChannel
from typing import Tuple
from .to_np import to_np
import numpy as np

def get_images_from_camera(channel_camera: StreamChannel, exporter: ZipkinExporter) -> Tuple[np.ndarray, Tracer, Span]:
    """Consumes the most recent image from a channel and prepares distributed tracing.

    Args:
        channel_camera (StreamChannel): The channel from which the image will be consumed.
        exporter (ZipkinExporter): The Zipkin exporter used to create the tracer.

    Returns:
        Tuple[np.ndarray, Tracer, Span]: The image as a NumPy array, the Tracer object, and the Span.
    """
    while True:
        message: Message = channel_camera.consume_last()
        if not isinstance(message, bool):
            continue
        tracer: Tracer = Tracer(
            exporter=exporter,
            span_context=message.extract_tracing()
        )
        span: Span = tracer.start_span(name="tiffany_detection")

        with tracer.span(name="get_and_unpack_image_from_camera"):
            image_proto = message.unpack(Image)
            image_np = to_np(image_proto)
            return image_np, tracer, span