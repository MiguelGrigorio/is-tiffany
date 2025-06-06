from classes import KeypointsDetector, Connection
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from is_wire.core import Logger, Message
from functions import get_images_from_camera, to_image
from is_wire.core import Logger
import time
import cv2
import os

def load_config():
    import json
    # Carrega configuração do ConfigMap (variável de ambiente CONFIG)
    config_str = os.getenv("CONFIG", "{}")
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        return {}

def download_model(download_link: str, log: Logger) -> None:
    from pathlib import Path
    import requests

    model_path = Path("models/best.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Cria o diretório se não existir
    # Verifica se o modelo já existe
    if not model_path.exists():
        log.info(f"Downloading model from {download_link} to {model_path}")
        response = requests.get(download_link)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            log.info(f"Model downloaded to {model_path}")
        else:
            log.critical(f"Failed to download model: {response.status_code}")
            raise RuntimeError(f"Failed to download model from {download_link}")

def main() -> None:
    config = load_config()
    
    # Configurações padrão podem vir do ConfigMap
    broker_uri = config.get("broker_uri", "amqp://guest:guest@10.10.2.211:30000")
    zipkin_uri = config.get("zipkin_uri", "http://10.10.2.211:30200")
    conf = float(config.get("CONF", "0.6"))  # Confidence pra pegar o bounding box

    camera_id = int(os.getenv("CAMERA_ID", "1"))  # ID da câmera do Deployment
    service_name = f"Tiffany.{camera_id}.Keypoints"
    
    log = Logger(name = service_name)
    download_model(config.get("model_link", "https://github.com/MiguelGrigorio/is-tiffany-keypoints-detector/raw/main/src/models/best.pt"), log)

    c = Connection(broker_uri, zipkin_uri, camera_id, service_name, log)
    channel_detection = c.channel_detection
    channel_camera = c.channel_camera
    exporter = c.exporter

    kp_detector = KeypointsDetector("models/best.pt", logger=log, device="cpu")

    while True:
        try:
            img, tracer, span, roi, original = get_images_from_camera(channel_camera, channel_detection, exporter, conf, camera_id)
        except KeyboardInterrupt:
            log.error("Closing...")
            raise(KeyboardInterrupt)
        except ConnectionResetError:
            log.warn("Timeout consume...")
            continue
        except IndexError:
            continue
        except OSError:
            log.warn("Resetting server...")
            c = Connection(broker_uri, zipkin_uri, camera_id, service_name, log)
            channel_detection = c.channel_detection
            channel_camera = c.channel_camera
            exporter = c.exporter
            continue

        if img.shape[0] * img.shape[1] < 3500:
            log.info(f"Small Bounding Box to Camera {camera_id}.")
            tracer.end_span()
            continue
    
        with tracer.span(name = "predict_keypoints_tiffany"):
            results = kp_detector.predict(img)
            result_dict = kp_detector.results_to_dict(results, roi)

        with tracer.span(name = "pack_and_publish_keypoints"):
            if len(result_dict["boxes"]):
                
                obj = ObjectAnnotations(
                    objects=[kp_detector.dict_to_obj_annot(result_dict)],
                    resolution=Resolution(height=720, width=1280),
                    frame_id=camera_id
                )
                predict_msg = Message()
                predict_msg.topic = f'Tiffany.{camera_id}.Keypoints'
                predict_msg.inject_tracing(span)
                predict_msg.pack(obj)
                predict_msg.created_at = time.time()

                channel_detection.publish(predict_msg)
            
                cv2.rectangle(original, (int(obj.objects[0].region.vertices[0].x), int(obj.objects[0].region.vertices[0].y)), (int(obj.objects[0].region.vertices[1].x), int(obj.objects[0].region.vertices[1].y)), (255, 0, 255), 1)
                cv2.circle(original, (int(obj.objects[0].keypoints[0].position.x), int(obj.objects[0].keypoints[0].position.y)), 2, (255, 0, 0), -1)
                cv2.circle(original, (int(obj.objects[0].keypoints[1].position.x), int(obj.objects[0].keypoints[1].position.y)), 2, (0, 0, 255), -1)
                image_msg = Message()
                image_msg.topic = f'TiffanyKeypoints.{camera_id}.Frame'
                image_msg.inject_tracing(span)
                image = to_image(original)
                image_msg.pack(image)
                image_msg.created_at = time.time()

                channel_camera.publish(image_msg)

                log.info(f'Published predicts for camera {camera_id}')

        tracer.end_span()


if __name__ == "__main__":
    main()
