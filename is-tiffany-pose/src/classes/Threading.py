from is_wire.core import Message, StatusCode, Status, Subscription, Channel
from is_msgs.common_pb2 import Pose, Position, Orientation
from google.protobuf.wrappers_pb2 import FloatValue
from is_msgs.image_pb2 import ObjectAnnotations
from functions import point2world, angle
from .AngleHistory import AngleHistory
from .Connection import Connection
import numpy as np
import threading
import time
import os

CONFIDENCE = os.environ.get("conf", 0.997)

class Threading:
    """
    Manages background threads for object keypoints detection and pose estimation.

    This class encapsulates the logic to run two main types of threads:
    1. Continuous detection threads that consume camera images and execute the model.
    2. On-demand streaming threads that annotate images with detection results
       and publish them for a specified duration.

    A lock is used to ensure thread-safe access to the latest detection data.
    """

    def __init__(self, connection: Connection, parameters: dict):
        """
        Initializes the thread manager.

        Args:
            connection (Connection): Object managing the broker connection.
            parameters (dict): Camera calibration parameters.
            log (Logger): Logger instance for recording messages.
        """
        self.angle = AngleHistory(max_history=10, max_age_seconds=10)
        self.connection = connection
        self.log = connection.log
        self.parameters = parameters
        self._last_keypoints = {}
        self._last_pose = Pose()
        self.keypoints_event = {cam_id: threading.Event() for cam_id in parameters.keys()}
        self.pose_event = threading.Event()
        self.lock = threading.Lock()

    def get_keypoints_by_camera(self, minutes: FloatValue, camera_id: int) -> None:
        """
        Continuously fetches keypoints from a specific camera for a defined duration.

        Args:
            minutes (FloatValue): Duration in minutes to fetch keypoints.
            camera_id (int): ID of the camera to fetch keypoints from.
        """
        self.keypoints_event[camera_id].set()

        duration_seconds = minutes.value * 60
        threading.current_thread().name = f"Keypoints.{camera_id}.Thread"
        start_time = time.time()
        self.log.info(f"Starting keypoints acquisition. Duration: {duration_seconds / 60:.2f} minutes.")
        
        channel = Channel(self.connection.broker_uri)
        subscription = Subscription(channel)
        
        while time.time() - start_time < duration_seconds:
            request = Message(reply_to=subscription)
            try:
                channel.publish(request, topic=f"Tiffany.Keypoints.{camera_id}.GetDetection")
                reply = channel.consume(timeout=1.0)
                if reply and reply.status.code == StatusCode.OK:
                    kp = reply.unpack(ObjectAnnotations)
                    if kp.objects and kp.objects[0].keypoints[0].score > CONFIDENCE and kp.objects[0].keypoints[1].score > CONFIDENCE:
                        self._last_keypoints[camera_id] = (kp, time.time())
            except:
                continue
        
        channel.close()
        self.log.info("Thread finished.")
        self.set_last_keypoints(None, camera_id)
        self.keypoints_event[camera_id].clear()

    def set_last_keypoints(self, keypoints: ObjectAnnotations, camera_id: int) -> None:
        """
        Updates the last keypoints safely for a specific camera.

        Args:
            keypoints (ObjectAnnotations): Last keypoints result.
            camera_id (int): Camera ID that provided the keypoints.
        """
        with self.lock:
            if keypoints is None:
                if camera_id in self._last_keypoints:
                    del self._last_keypoints[camera_id]
            else:
                self._last_keypoints[camera_id] = (keypoints, time.time())

    def set_last_pose(self, pose: Pose) -> None:
        """
        Updates the last detected pose safely.

        Args:
            pose (Pose): The latest detected pose.
        """
        with self.lock:
            self._last_pose = pose

    def get_last_pose(self, *args) -> Pose:
        """
        Retrieves the last detected pose safely.

        Returns:
            Pose: Last stored detected pose.
        """
        with self.lock:
            return self._last_pose

    def get_last_keypoints(self) -> dict:
        """
        Retrieves the last keypoints results safely.

        Returns:
            Dict[int, ObjectAnnotations]: Last stored keypoints for each camera.
        """
        with self.lock:
            return self._last_keypoints

    def define_pose(self, minutes: FloatValue) -> None:
        """Continuously calculates and updates Tiffany's pose using keypoints from multiple cameras.

        Args:
            minutes (FloatValue): Duration to compute the pose in minutes.
        """
        self.pose_event.set()
        duration_seconds = minutes.value * 60
        threading.current_thread().name = "PoseThread"
        start_time = time.time()
        self.log.info(f"Starting pose calculation for {duration_seconds / 60:.2f} minutes.")
        last_pose_time = 0.0
        while time.time() - start_time < duration_seconds:
            if time.time() - last_pose_time > 5.0:
                self.set_last_pose(Pose())
            keypoints = self.get_last_keypoints()
            if len(keypoints) < 2:
                continue  # Need at least two cameras

            # Extract center and front points from recent keypoints
            kp_center = {cam_id: (kp.objects[0].keypoints[0].position.x,
                                kp.objects[0].keypoints[0].position.y)
                        for cam_id, (kp, ts) in keypoints.items()
                        if time.time() - ts < 5.0}
            kp_front = {cam_id: (kp.objects[0].keypoints[1].position.x,
                                kp.objects[0].keypoints[1].position.y)
                        for cam_id, (kp, ts) in keypoints.items()
                        if time.time() - ts < 5.0}

            if len(kp_center) < 2 or len(kp_front) < 2:
                continue

            # Convert image points to world coordinates
            Xw_center = np.round(point2world(self.parameters, kp_center), 3)
            Xw_front = np.round(point2world(self.parameters, kp_front), 3)

            # Compute vector and angle
            vTiffany = Xw_front[:2] - Xw_center[:2]
            _, angle_deg, _ = self.angle.add_and_check(angle(np.array([1, 0]), vTiffany), timestamp=time.time())

            # Update pose
            pose = Pose(
                position=Position(x=Xw_center[0], y=Xw_center[1], z=Xw_center[2]),
                orientation=Orientation(yaw=angle_deg)
            )
            self.set_last_pose(pose)
            last_pose_time = time.time()

        self.log.info("Pose calculation finished.")
        self.set_last_pose(Pose())
        self.pose_event.clear()


    def start_detections(self, minutes: FloatValue, ctx) -> Status:
        """
        Starts the detection thread if not already running.

        Args:
            minutes (FloatValue): Duration of detection in minutes.
            ctx: RPC service context provided by is-wire.

        Returns:
            Status: OK if detection started, ALREADY_EXISTS if detection is
                    already running, DEADLINE_EXCEEDED if the detection service
                    did not respond.
        """
        if any(event.is_set() for event in self.keypoints_event.values()) or self.pose_event.is_set():
            return Status(StatusCode.ALREADY_EXISTS, 'Detection already in progress')
        for cam_id in self.parameters.keys():
            channel = Channel(self.connection.broker_uri)
            subscription = Subscription(channel)
            request = Message(content=FloatValue(value=minutes.value + 1), reply_to=subscription)
            try:
                channel.publish(request, topic=f"Tiffany.Keypoints.{cam_id}.StartDetection")
                reply = channel.consume(timeout=5.0)
                time.sleep(0.5)
                if reply.status.code in [StatusCode.OK, StatusCode.ALREADY_EXISTS]:
                    if not self.keypoints_event[cam_id].is_set():
                        thread = threading.Thread(target=self.get_keypoints_by_camera, args=(minutes, cam_id))
                        thread.daemon = True
                        thread.start()
            except:
                return Status(StatusCode.DEADLINE_EXCEEDED, 'No response from detection service')
        if not self.pose_event.is_set():
            pose_thread = threading.Thread(target=self.define_pose, args=(minutes,))
            pose_thread.daemon = True
            pose_thread.start()
        return Status(StatusCode.OK, 'Detections started successfully')