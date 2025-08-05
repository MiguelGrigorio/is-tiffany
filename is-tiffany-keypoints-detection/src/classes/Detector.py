from is_msgs.image_pb2 import ObjectAnnotation, BoundingPoly, Vertex, PointAnnotation
from ultralytics.engine.results import Results
from typing import List, Dict, Any
from is_wire.core import Logger
from ultralytics import YOLO
import numpy as np

class Detector():
    def __init__(self, model_path: str, logger: Logger, device: str = "cpu") -> None:
        self.model: YOLO = YOLO(model_path)
        self.model.to(device)
        logger.info(f"Loaded model: {model_path}")

    def predict(self, img: np.ndarray) -> Results:
        results = self.model.predict(source=img, imgsz=96, verbose = False)
        return results[0]

    def results_to_dict(self, results: Results, offset: np.ndarray) -> Dict[str, List[dict]]:
        '''
        Converte os resultados de detecção de keypoints do modelo YOLO para um dicionário.

        Parâmetros:
            results (Results): Objeto de resultados da detecção do Ultralytics YOLO.
            offset (np.ndarray): Deslocamento a ser aplicado às coordenadas dos keypoints.

        Retorna:
            dict: Um dicionário com duas chaves:
                - "boxes": lista de dicionários contendo a confiança ("conf") e coordenadas "xyxy".
                - "keypoints": lista de dicionários contendo a confiança ("conf") e coordenadas "xy" para cada ponto-chave.
        '''
        results_dict = {
            "boxes": [],
            "keypoints": [],
        }
        offset_4_x_1 = np.tile(offset, 2)
        num_results = len(results.boxes)
        if num_results > 0:
            results_dict["boxes"].append({
                "conf": results.boxes.conf.numpy()[0],
                "xyxy": results.boxes.xyxy.numpy()[0] + offset_4_x_1,
            })
            results_dict["keypoints"].append({
                "conf": results.keypoints.conf.numpy()[0][0],
                "xy": results.keypoints.xy.numpy()[0][0] + offset,
            })
            results_dict["keypoints"].append({
                "conf": results.keypoints.conf.numpy()[0][1],
                "xy": results.keypoints.xy.numpy()[0][1] + offset,
            })
        return results_dict

    @staticmethod
    def dict_to_obj_annot(result_dict: Dict[str, Any]) -> ObjectAnnotation:
        '''
        Converte um dicionário contendo bounding box e keypoints no formato Tiffany
        para um objeto `ObjectAnnotation` do protocolo IS.

        Parâmetros:
            result_dict (dict): Dicionário com a chave 'boxes' e 'keypoints'.

        Retorna:
            ObjectAnnotation: Anotação com rótulo, score, região e pontos-chave.
        '''
        box = result_dict["boxes"][0]
        kps = result_dict["keypoints"]
        conf = (float(100*(kps[0]["conf"]-0.99)), float(100*(kps[1]["conf"]-0.99))) # Normalizando a partir da terceira casa decimal
        
        return ObjectAnnotation(
            label="Tiffany",
            id=0,
            score=box["conf"],
            region=BoundingPoly(
                vertices=[
                    Vertex(x=box["xyxy"][0], y=box["xyxy"][1]),
                    Vertex(x=box["xyxy"][2], y=box["xyxy"][3]),
                ]
            ),
            keypoints=[
                PointAnnotation(
                    id=i,
                    score=conf[i],
                    position=Vertex(x=kps[i]["xy"][0], y=kps[i]["xy"][1]),
                )
                for i in range(len(kps))
            ]
        )