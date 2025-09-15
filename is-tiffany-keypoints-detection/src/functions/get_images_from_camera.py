from is_wire.core import Tracer, Message, Subscription, Channel
from is_msgs.image_pb2 import Image, ObjectAnnotations
from opencensus.trace.blank_span import BlankSpan
from classes import Connection, StreamChannel
from typing import Tuple
from .to_np import to_np
import numpy as np
import time
import os

CONFIDENCE = float(os.environ.get("confidence", 0.5))

def get_images_from_camera(channel_camera: StreamChannel, connection: Connection, end_time: float) -> Tuple[np.ndarray, Tracer, BlankSpan]:
    '''
    Obtains the cropped image (ROI) from the camera detection.

    Args:
        channel_camera (StreamChannel): StreamChannel object for consuming camera frames.
        connection (Connection): Connection object containing the channels and the exporter.
        end_time (float): The time at which the function should stop trying to get images.
    
    Returns:
        Tuple containing:
            - crop (np.ndarray): Cropped region of interest (ROI) image from the camera.
            - tracer (Tracer): Tracer object for distributed monitoring.
            - span (BlankSpan): Active trace span for the operation.
            - roi_offset (np.ndarray): Coordinates (x1, y1) of the top-left corner of the ROI in the original image.
            - original_img (np.ndarray): Full original image from the camera.
    '''
    channel_detection = Channel(connection.broker_uri)
    exporter = connection.exporter
    camera_id = connection.camera_id

    subscription = Subscription(channel_detection)
    request = Message(reply_to=subscription)

    while time.time() < end_time:
        try:
            channel_detection.publish(request, topic=f"Tiffany.Detection.{camera_id}.GetDetection")
            reply = channel_detection.consume(timeout = 1.0)
            det = reply.unpack(ObjectAnnotations)
        except:
            continue
        
        if det.objects:
            score: float = det.objects[0].score
            if score < CONFIDENCE:
                continue
            box = det.objects[0].region.vertices
            x1, y1 = int(box[0].x), int(box[0].y)
            x2, y2 = int(box[1].x), int(box[1].y)
            if (x2 - x1) * (y2 - y1) < 3500:
                continue
            tracer: Tracer = Tracer(
                exporter=exporter,
                span_context=reply.extract_tracing()
            )
            span: BlankSpan = tracer.start_span(name="tiffany_keypoints_detection")

            with tracer.span(name="get_and_unpack_image_from_camera"):
                while time.time() < end_time:
                    image = channel_camera.consume_last()
                    if not isinstance(image, bool):
                        img = image.unpack(Image)
                        original_img = to_np(img)
                        
                        crop = original_img[y1:y2, x1:x2]
                        roi_offset = np.array([x1, y1])
                        channel_detection.close()
                        return crop, tracer, span, roi_offset, original_img