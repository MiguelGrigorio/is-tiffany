from is_wire.core import Message, StatusCode, Status, Subscription
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from google.protobuf.wrappers_pb2 import FloatValue
from opencensus.trace.blank_span import BlankSpan
from amqp.exceptions import UnexpectedFrame
from .StreamChannel import StreamChannel
from opencensus.trace.span import Span
from typing import Optional, Union
from .Connection import Connection
from .Detector import Detector
import numpy as np
import threading
import time
import cv2


class Threading:
    """Manages background threads for object detection and streaming results.

    This class encapsulates logic to run two main threads:
    1. A continuous detection thread that consumes images and runs the model.
    2. An on-demand streaming thread that annotates images with detections and
       publishes them for a specified duration.

    A lock is used to ensure thread-safe access to the last detection data.
    """

    def __init__(self, connection: Connection, detector: Detector):
        """Initializes the threading manager.

        Args:
            connection (Connection): Manages the broker connection.
            detector (Detector): Responsible for running predictions.
        """
        self.connection = connection
        self.log = connection.log
        self.detector = detector

        self._last_detection: ObjectAnnotations = ObjectAnnotations()
        self._last_span: Union[Span, BlankSpan] = BlankSpan()
        self._last_image: Optional[np.ndarray] = None

        self.stream_event = threading.Event()
        self.detection_event = threading.Event()
        self.lock = threading.Lock()

    def detection_thread(self, minutes: FloatValue) -> None:
        """Runs for a defined duration to fetch images and perform detection.

        Designed to run as a daemon thread in the background. It consumes images
        from the camera feed, runs detection, and stores the latest results safely.
        Handles connection errors by attempting to reset when needed.
        """
        from functions import get_images_from_camera
        self.detection_event.set()

        channel_camera = StreamChannel(self.connection.broker_uri)
        Subscription(channel_camera).subscribe(f"CameraGateway.{self.connection.camera_id}.Frame")
        exporter = self.connection.exporter

        threading.current_thread().name = "DetectionThread"

        duration_seconds = minutes.value * 60
        start_time = time.time()
        self.log.info(f"Detection started. Duration: {duration_seconds / 60:.2f} minutes.")

        while time.time() - start_time < duration_seconds:
            try:
                img, tracer, span = get_images_from_camera(channel_camera, exporter)
            except KeyboardInterrupt:
                self.log.error("Shutting down...")
                raise
            except (ConnectionResetError, IndexError, UnexpectedFrame):
                self.log.warn("Skipping frame due to temporary issue.")
                continue
            except OSError:
                self.log.warn("Resetting server connection due to OSError...")
                time.sleep(2.5)
                channel_camera = StreamChannel(self.connection.broker_uri)
                Subscription(channel_camera).subscribe(f"CameraGateway.{self.connection.camera_id}.Frame")
                continue

            with tracer.span(name="predict_tiffany"):
                results = self.detector.predict(img)
                result_dict = self.detector.results_to_dict(results)

            with tracer.span(name="pack_and_publish_detection"):
                if len(result_dict["boxes"]):
                    obj = ObjectAnnotations(
                        objects=[self.detector.dict_to_obj_annot(result_dict)],
                        resolution=Resolution(height=720, width=1280),
                        frame_id=self.connection.camera_id
                    )
                    self.set_last_detection_and_image_and_span(obj, img, span)

            tracer.end_span()
        channel_camera.close()
        self.log.info("Detection finished.")
        self.set_last_detection_and_image_and_span(ObjectAnnotations(), None, BlankSpan())
        self.detection_event.clear()

    def set_last_detection_and_image_and_span(
        self, 
        detection: ObjectAnnotations, 
        image: Optional[np.ndarray], 
        span: Union[Span, BlankSpan]
    ) -> None:
        """Safely updates the last detection, image, and tracing span.

        Args:
            detection (ObjectAnnotations): Detected object annotations.
            image (Optional[np.ndarray]): The image on which detection was performed.
            span (Union[Span, BlankSpan]): The tracing span associated with the detection.
        """
        with self.lock:
            self._last_detection = detection
            self._last_image = image
            self._last_span = span

    def get_last_detection(self) -> ObjectAnnotations:
        """Safely retrieves the latest detection result.

        Returns:
            ObjectAnnotations: The most recent detection stored.
        """
        with self.lock:
            return self._last_detection

    def get_last_image(self) -> Optional[np.ndarray]:
        """Safely retrieves the latest image with detection.

        Returns:
            Optional[np.ndarray]: The last stored image, or None if none exists.
        """
        with self.lock:
            return self._last_image

    def get_last_span(self) -> Union[Span, BlankSpan]:
        """Safely retrieves the last tracing span.

        Returns:
            Union[Span, BlankSpan]: The last span associated with a detection.
        """
        with self.lock:
            return self._last_span

    def stream_detection_thread(self, minutes: FloatValue) -> None:
        """Draws detections on images and streams them for a defined duration.

        Runs in a separate thread. Continuously fetches the latest detection,
        draws bounding boxes on a copy of the image, and publishes it to a topic.
        Terminates after the specified duration.

        Args:
            minutes (FloatValue): Duration in minutes for streaming.
        """
        from functions import to_image

        self.stream_event.set()
        threading.current_thread().name = "StreamThread"
        channel = StreamChannel(self.connection.broker_uri)
        duration_seconds = minutes.value * 60
        init_time = time.time()
        self.log.info(f"Streaming started. Duration: {duration_seconds / 60:.2f} minutes.")

        while time.time() - init_time < duration_seconds:
            det = self.get_last_detection()
            img = self.get_last_image()
            span = self.get_last_span()

            if img is None:
                continue

            img_to_draw = img.copy()

            if det.objects:
                box = det.objects[0].region.vertices
                bb1 = (int(box[0].x), int(box[0].y))
                bb2 = (int(box[1].x), int(box[1].y))
                cv2.rectangle(img_to_draw, bb1, bb2, (255, 255, 0), 2)
                cv2.putText(
                    img_to_draw, 
                    f"Score: {det.objects[0].score:.2f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 0), 
                    2
                )

            try:
                msg = Message()
                msg.inject_tracing(span)
                msg.topic = f"Tiffany.{self.connection.camera_id}.Frame"
                msg.pack(to_image(img_to_draw))
                channel.publish(msg)
            except (UnexpectedFrame, ConnectionResetError, OSError):
                self.log.warn("Resetting publishing connection...")
                time.sleep(2.5)
                self.connection.reset_connection()
                time.sleep(2.5)
            except Exception as e:
                self.log.error(f"Unexpected error while publishing: {e}")
                continue
        
        channel.close()
        self.log.info("Streaming finished.")
        self.stream_event.clear()

    def init_stream(self, minutes: FloatValue, ctx) -> Status:
        """Starts the detection streaming thread if not already running.

        Exposed as an RPC method. Checks if a stream is active using an event flag,
        and starts a new `stream_detection_thread` if none is running.

        Args:
            minutes (FloatValue): Desired duration of the stream in minutes.
            ctx: Service context provided by is-wire RPC.

        Returns:
            Status: `OK` if the stream started, or `ALREADY_EXISTS` if one is already running.
        """
        if not self.stream_event.is_set():
            if not self.detection_event.is_set():
                self.init_detection(FloatValue(value=minutes.value + 1), ctx)
                time.sleep(1.0)
            thread = threading.Thread(target=self.stream_detection_thread, args=(minutes,))
            thread.daemon = True
            thread.start()
            return Status(StatusCode.OK, "Stream started")
        else:
            return Status(StatusCode.ALREADY_EXISTS, "Stream already running")

    def init_detection(self, minutes: FloatValue, ctx) -> Status:
        """Starts the detection thread if not already running.

        Exposed as an RPC method. Checks if the detection thread is active using
        an event flag, and starts a new `detection_thread` if none is running.

        Args:
            minutes (FloatValue): Desired duration of detection in minutes.
            ctx: Service context provided by is-wire RPC.

        Returns:
            Status: `OK` if detection started, or `ALREADY_EXISTS` if already running.
        """
        if not self.detection_event.is_set():
            thread = threading.Thread(target=self.detection_thread, args=(minutes,))
            thread.daemon = True
            thread.start()
            return Status(StatusCode.OK, "Detection started")
        else:
            return Status(StatusCode.ALREADY_EXISTS, "Detection already running")
