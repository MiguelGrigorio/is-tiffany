from is_wire.core import Message, StatusCode, Status, Subscription, Channel
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from google.protobuf.wrappers_pb2 import FloatValue
from opencensus.trace.blank_span import BlankSpan
from amqp.exceptions import UnexpectedFrame
from .StreamChannel import StreamChannel
from opencensus.trace.span import Span
from .Connection import Connection
from .Detector import Detector
import numpy as np
import threading
import socket
import time
import cv2


class Threading:
    """Manages background threads for object detection and result streaming.

    This class encapsulates the logic to run two main threads:
    1. A continuous detection thread that consumes images and runs the model.
    2. An on-demand streaming thread that annotates images with detections
       and publishes them for a specified duration.

    A lock is used to ensure thread-safe access to the last detection data.
    """

    def __init__(self, connection: Connection, detector: Detector):
        """Initializes the threading manager.

        Args:
            connection (Connection): An object that manages the broker connection.
            detector (Detector): An object responsible for running predictions.
        """
        self.connection = connection
        self.log = connection.log
        self.detector = detector
        self._last_detection = ObjectAnnotations()
        self._last_span: Span | BlankSpan = BlankSpan()
        self._last_image: np.ndarray | None = None
        self.stream_event = threading.Event()
        self.detection_event = threading.Event()
        self.lock = threading.Lock()

    def detection_thread(self, minutes: FloatValue) -> None:
        """Runs for a defined duration to fetch images and perform detection.

        Designed to run as a daemon thread in the background. It consumes images
        from the camera feed, passes them to the detector, and safely stores
        the most recent results. Also handles connection errors, attempting
        to reset the connection when needed.
        """
        from functions import get_images_from_camera
        self.detection_event.set()

        channel_camera = StreamChannel(self.connection.broker_uri)
        Subscription(channel_camera).subscribe(f"CameraGateway.{self.connection.camera_id}.Frame")
        duration_seconds = minutes.value * 60
        threading.current_thread().name = "DetectionThread"
        start_time = time.time()
        self.log.info(f"Detection started. Duration: {duration_seconds / 60:.2f} minutes.")
        end_time = start_time + duration_seconds
        while time.time() < end_time:
            try:
                img, tracer, span, offset, original_img = get_images_from_camera(channel_camera, self.connection, end_time)

            except KeyboardInterrupt:
                self.log.error("Shutting down...")
                raise KeyboardInterrupt

            except (ConnectionResetError, IndexError, UnexpectedFrame, TypeError):
                #self.log.warn("Skipping frame due to temporary issue.")
                continue

            except OSError:
                self.log.warn("Restarting server connection due to OSError...")
                time.sleep(2.5)
                channel_camera = StreamChannel(self.connection.broker_uri)
                Subscription(channel_camera).subscribe(f"CameraGateway.{self.connection.camera_id}.Frame")
                continue

            with tracer.span(name="predict_tiffany"):
                results = self.detector.predict(img)
                result_dict = self.detector.results_to_dict(results, offset)

            with tracer.span(name="pack_and_publish_detection"):
                if len(result_dict["boxes"]):
                    obj = ObjectAnnotations(
                        objects=[self.detector.dict_to_obj_annot(result_dict)],
                        resolution=Resolution(height=720, width=1280),
                        frame_id=self.connection.camera_id
                    )
                    self.set_last_detection_and_image_and_span(obj, original_img, span)

            tracer.end_span()
        
        self.log.info("Detection finished.")
        self.set_last_detection_and_image_and_span(ObjectAnnotations(), None, BlankSpan())
        self.detection_event.clear()

    def set_last_detection_and_image_and_span(
        self, detection: ObjectAnnotations, image: np.ndarray, span: Span | BlankSpan
    ) -> None:
        """Safely updates the last detection, image, and tracing span.

        Args:
            detection (ObjectAnnotations): Detected object annotations.
            image (np.ndarray): The image on which detection was performed.
            span (Span | BlankSpan): The tracing span associated with the detection.
        """
        with self.lock:
            self._last_detection = detection
            self._last_image = image
            self._last_span = span

    def get_last_detection(self, *args) -> ObjectAnnotations:
        """Safely retrieves the latest detection result.

        Returns:
            ObjectAnnotations: The most recent detection stored.
        """
        with self.lock:
            return self._last_detection
        
    def get_last_image(self) -> np.ndarray | None:
        """Safely retrieves the latest image with detection.

        Returns:
            np.ndarray | None: The last stored image, or None if none exists.
        """
        with self.lock:
            return self._last_image
            
    def get_last_span(self) -> Span | BlankSpan:
        """Safely retrieves the last tracing span.

        Returns:
            Span | BlankSpan: The last span associated with a detection.
        """
        with self.lock:
            return self._last_span

    def stream_detection_thread(self, minutes: FloatValue) -> None:
        """Draws detections on images and streams them for a defined duration.

        Runs in a separate thread. Continuously fetches the latest detection,
        draws bounding boxes and keypoints on a copy of the image, and publishes
        it to a topic. Terminates after the specified duration.

        Args:
            minutes (FloatValue): Duration in minutes for streaming.
        """
        from functions import to_image
        self.stream_event.set()
        init_time = time.time()
        threading.current_thread().name = "StreamThread"
        duration_seconds = minutes.value * 60
        self.log.info(f"Streaming started. Duration: {duration_seconds / 60:.2f} minutes.")
        channel = Channel(self.connection.broker_uri)
        
        while time.time() - init_time < duration_seconds:
            det = self.get_last_detection()
            img = self.get_last_image()
            span = self.get_last_span()

            if img is None:
                continue
            
            img_to_draw = img.copy()

            if det.objects:
                kp = det.objects[0].keypoints

                kp1 = [kp[0].position.x, kp[0].position.y]
                kp2 = [kp[1].position.x, kp[1].position.y]
                box = det.objects[0].region.vertices
                bb1 = (int(box[0].x), int(box[0].y))
                bb2 = (int(box[1].x), int(box[1].y))

                cv2.rectangle(img_to_draw, bb1, bb2, (255, 255, 0), 2)
                cv2.circle(img, (int(kp1[0]), int(kp1[1])), 3, (0, 255, 0), -1)
                cv2.circle(img, (int(kp2[0]), int(kp2[1])), 3, (0, 0, 255), -1)
                cv2.putText(img, f"{kp[0].score:.2f} | {(kp[0].score - 0.99)*100}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f"{kp[1].score:.2f} | {(kp[0].score - 0.99)*100}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, f"{det.objects[0].score:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                try:
                    msg = Message()
                    msg.inject_tracing(span)
                    msg.topic = f"Tiffany.Keypoints.{self.connection.camera_id}.Frame"
                    msg.pack(to_image(img_to_draw))
                    channel.publish(msg)
                except (UnexpectedFrame, ConnectionResetError, OSError):
                    self.log.warn("Restarting publishing connection...")
                    time.sleep(2.5)
                    channel = Channel(self.connection.broker_uri)
                    time.sleep(2.5)
                except Exception as e:
                    self.log.error(f"Unexpected error while publishing: {e}")
                    continue
        
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
            channel = Channel(self.connection.broker_uri)
            subscription = Subscription(channel)
            request = Message(content=FloatValue(value=minutes.value + 1), reply_to=subscription)
            channel.publish(request, topic=f"Tiffany.Detection.{self.connection.camera_id}.StartDetection")
            try:
                reply = channel.consume(timeout=5.0)
            except socket.timeout:
                return Status(StatusCode.DEADLINE_EXCEEDED, 'No response from detection service')
            if reply.status.code == StatusCode.OK or reply.status.code == StatusCode.ALREADY_EXISTS:
                thread = threading.Thread(target=self.detection_thread, args=(minutes,))
                thread.daemon = True
                thread.start()
            channel.close()
            return Status(StatusCode.OK, "Detection started")
        else:
            return Status(StatusCode.ALREADY_EXISTS, "Detection already running")