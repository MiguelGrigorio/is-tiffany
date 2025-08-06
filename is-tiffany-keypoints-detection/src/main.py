from classes import Detector, Connection
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from is_wire.core import Logger, Message
from functions import get_images_from_camera
from is_wire.core import Logger
import time
import os


def load_config():
    import json
    # Carrega configuração do ConfigMap (variável de ambiente CONFIG)
    config_str = os.getenv("CONFIG", "{}")
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        return {}

def download_model(model_url: str, model_path: str, log: Logger) -> None:
    import requests
    from pathlib import Path

    # Cria o diretório se não existir
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Faz o download do modelo
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        log.info(f"Downloaded: {model_path}")
    else:
        log.error(f"Error downloading model: {response.status_code}, retrying...")
        download_model(model_url, model_path, log)

def main() -> None:
    config = load_config()
    
    # Configurações padrão podem vir do ConfigMap
    broker_uri = config.get("broker_uri", "amqp://guest:guest@10.10.2.211:30000")
    zipkin_uri = config.get("zipkin_uri", "http://10.10.2.211:30200")
    model_url = config.get("model_url", "https://github.com/MiguelGrigorio/is-tiffany/raw/refs/heads/main/is-tiffany-keypoints-detection/src/models/orientation_model.pt")
    conf = config.get("conf", 0.7)

    camera_id = int(os.getenv("CAMERA_ID", 1))
    
    service_name = f"Tiffany.{camera_id}.KeypointsDetector"
    
    log = Logger(name = service_name)

    log.info("Verifying model...")
    download_model(model_url, "models/orientation_model.pt", log)

    c = Connection(broker_uri, zipkin_uri, camera_id, service_name, log)
    channel_camera = c.channel_camera
    channel_detection = c.channel_detection
    exporter = c.exporter

    detector = Detector("models/orientation_model.pt", logger=log, device="cpu")

    while True:
        try:
            crop, tracer, span, roi_offset = get_images_from_camera(channel_camera, channel_detection, exporter, conf)
            if crop.shape[0] * crop.shape[1] < 3500:
                log.info(f"Small Bounding Box to Camera {camera_id}.")
                tracer.end_span()
                continue
        except KeyboardInterrupt:
            log.error("Closing...")
            raise(KeyboardInterrupt)
        except ConnectionResetError:
            log.warn("Timeout consume...")
            continue
        except IndexError:
            continue
        except OSError as e:
            print(f"Error: {e}")
            log.warn(f"Resetting server...")
            time.sleep(5)
            c = Connection(broker_uri, zipkin_uri, camera_id, service_name, log)
            channel_camera = c.channel_camera
            channel_detection = c.channel_detection
            exporter = c.exporter
            continue

        with tracer.span(name = "predict_keypoints_tiffany"):
            results = detector.predict(crop)
            result_dict = detector.results_to_dict(results, roi_offset)

        with tracer.span(name = "pack_and_publish_detection"):
            if len(result_dict["boxes"]):
                
                obj = ObjectAnnotations(
                    objects=[detector.dict_to_obj_annot(result_dict)],
                    resolution=Resolution(height=720, width=1280),
                    frame_id=camera_id
                )
                predict_msg = Message()
                predict_msg.topic = f'Tiffany.{camera_id}.Keypoints'
                predict_msg.inject_tracing(span)
                predict_msg.pack(obj)
                predict_msg.created_at = time.time()
                channel_camera.publish(predict_msg)
                log.info(f"Keypoints detected in camera {camera_id} with confidence {100*(result_dict['keypoints'][0]['conf'] - 0.99):.2f} and {100*(result_dict['keypoints'][1]['conf'] - 0.99):.2f}.")
            else:
                log.info(f"No keypoints detected on camera {camera_id}.")
        tracer.end_span()


if __name__ == "__main__":
    main()
