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
                

camera_id = 4
choose = 'cv2'
channel_camera = StreamChannel()
channel_detection = StreamChannel()

Subscription(channel_camera).subscribe(f"CameraGateway.{camera_id}.Frame")
Subscription(channel_detection).subscribe(f"Tiffany.{camera_id}.Detection")
while True:
    img, det = get_images_from_camera(channel_camera, channel_detection, conf=0.8)
    box = det.objects[0].region.vertices
    bb1 = (int(box[0].x), int(box[0].y))
    bb2 = (int(box[1].x), int(box[1].y))
    cv2.rectangle(img, bb1, bb2, (255, 255, 0), 2)
    cv2.putText(img, f"Score: {det.objects[0].score:.2f}", (bb1[0], bb1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    if choose == 'pub':
        msg = Message(content = to_image(img))
        channel_camera.publish(msg, f"Tiffany.{camera_id}.Frame")
    else:
        cv2.imshow(f"Camera {camera_id}", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break