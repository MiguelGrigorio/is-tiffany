from is_wire.core import Channel, Message, Subscription
from is_msgs.image_pb2 import Image, ObjectAnnotations
import numpy as np
import socket
import cv2

class StreamChannel(Channel):
    def __init__(self, uri: str = "amqp://guest:guest@10.10.2.211:30000", exchange: str = "is"):
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self, return_dropped: bool = False):
        dropped = 0
        msg = super().consume()
        while True:
            try:
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped) if return_dropped else msg


def to_np(input_image) -> np.ndarray:
    if isinstance(input_image, np.ndarray):
        return input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
        return output_image if output_image is not None else np.array([], dtype=np.uint8)
    else:
        return np.array([], dtype=np.uint8)
    

def to_image(image, encode_format: str = ".jpeg", compression_level: float = 0.8) -> Image:
        if encode_format == ".jpeg":
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()
        cimage = cv2.imencode(ext=encode_format, img=image, params=params)
        return Image(data=cimage[1].tobytes())


def get_images_from_camera(channel_camera: StreamChannel, channel_detection: StreamChannel, conf: float):
    while True:
        obj_det = channel_detection.consume_last()

        if not isinstance(obj_det, bool):
            det = obj_det.unpack(ObjectAnnotations)
            score: float = det.objects[0].score
            if score < conf:
                continue
            while True:
                image = channel_camera.consume_last()

                if not isinstance(image, bool):
                    img = image.unpack(Image)
                    img_np = to_np(img)
                    return img_np, det
                

camera_id = 3
choose = 'cv2'
channel_camera = StreamChannel()
channel_detection = StreamChannel()

Subscription(channel_camera).subscribe(f"CameraGateway.{camera_id}.Frame")
Subscription(channel_detection).subscribe(f"Tiffany.{camera_id}.Keypoints")
while True:
    img, det = get_images_from_camera(channel_camera, channel_detection, conf=0.8)
   
    kp = det.objects[0].keypoints
    if kp[0].score < 0.6 or kp[1].score < 0.6:
        continue

    kp1 = [kp[0].position.x, kp[0].position.y]
    kp2 = [kp[1].position.x, kp[1].position.y]

    cv2.circle(img, (int(kp1[0]), int(kp1[1])), 3, (0, 255, 0), -1)
    cv2.circle(img, (int(kp2[0]), int(kp2[1])), 3, (0, 0, 255), -1)
    cv2.putText(img, f"{kp[0].score:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f"{kp[1].score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if choose == 'pub':
        msg = Message(content = to_image(img))
        channel_camera.publish(msg, f"Tiffany.{camera_id}.Frame")
    else:
        cv2.imshow(f"Camera {camera_id}", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break