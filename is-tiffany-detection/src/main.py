from classes import Detector, Connection
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from is_wire.core import Logger, Message
from functions import get_images_from_camera
from is_wire.core import Logger
import time
import os
import re


def load_config():
    import json
    # Carrega configuração do ConfigMap (variável de ambiente CONFIG)
    config_str = os.getenv("CONFIG", "{}")
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        return {}

def download_model(model_url: str, model_path: str) -> None:
    import requests
    from pathlib import Path

    # Cria o diretório se não existir
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Faz o download do modelo
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Modelo baixado com sucesso: {model_path}")
    else:
        print(f"Erro ao baixar o modelo: {response.status_code}")

def main() -> None:
    config = load_config()
    
    # Configurações padrão podem vir do ConfigMap
    broker_uri = config.get("broker_uri", "amqp://guest:guest@10.10.2.211:30000")
    zipkin_uri = config.get("zipkin_uri", "http://10.10.2.211:30200")
    model_url = config.get("model_url", "https://github.com/MiguelGrigorio/is-tiffany/raw/refs/heads/main/is-tiffany-detection/src/models/detection_model.pt")

    download_model(model_url, "models/detection_model.pt")
    
    camera_id = int(os.getenv("CAMERA_ID", 1))

    service_name = f"Tiffany.{camera_id}.Detection"
    
    log = Logger(name = service_name)

    c = Connection(broker_uri, zipkin_uri, camera_id, service_name, log)
    channel_camera = c.channel_camera
    exporter = c.exporter

    detector = Detector("models/detection_model.pt", logger=log, device="cuda")

    while True:
        try:
            img, tracer, span = get_images_from_camera(channel_camera, exporter)
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
            channel_camera = c.channel_camera
            exporter = c.exporter
            continue
    
        with tracer.span(name = "predict_tiffany"):
            results = detector.predict(img)
            result_dict = detector.results_to_dict(results)

        with tracer.span(name = "pack_and_publish_detection"):
            if len(result_dict["boxes"]):
                
                obj = ObjectAnnotations(
                    objects=[detector.dict_to_obj_annot(result_dict)],
                    resolution=Resolution(height=720, width=1280),
                    frame_id=camera_id
                )
                predict_msg = Message()
                predict_msg.topic = f'Tiffany.{camera_id}.Detection'
                predict_msg.inject_tracing(span)
                predict_msg.pack(obj)
                predict_msg.created_at = time.time()
                channel_camera.publish(predict_msg)
                log.info(f"Tiffany detectada na câmera {camera_id} com confiança {result_dict['boxes'][0]['conf']:.2f}.")
            else:
                log.info(f"Nenhuma detecção da Tiffany na câmera {camera_id}.")
        tracer.end_span()


if __name__ == "__main__":
    main()
