from is_msgs.image_pb2 import ObjectAnnotation, BoundingPoly, Vertex
from ultralytics.engine.results import Results
from typing import List, Dict, Any
from is_wire.core import Logger
from ultralytics import YOLO
import numpy as np

class Detector():
    def __init__(self, model_path: str, logger: Logger, device: str = "cuda") -> None:
        self.model: YOLO = YOLO(model_path)
        self.model.to(device)
        logger.info(f"Loaded model: {model_path}")

    def predict(self, img: np.ndarray) -> Results:
        results = self.model.predict(source=img, imgsz=640, verbose = False)
        return results[0]
    
    def results_to_dict(self, results: Results) -> Dict[str, List[dict]]:
        '''
        Converte os resultados de detecção do modelo YOLO para um dicionário.

        Parâmetros:
            results (Results): Objeto de resultados da detecção do Ultralytics YOLO.

        Retorna:
            dict: Um dicionário com duas chaves:
                - "boxes": lista de dicionários contendo a confiança ("conf") e coordenadas "xyxy".
        '''
        results_dict = {
            "boxes": [],
        }
        num_results = len(results.boxes)
        if num_results > 0:
            results_dict["boxes"].append({
                "conf": results.boxes.conf.numpy()[0],
                "xyxy": results.boxes.xyxy.numpy()[0],
                
            })
        return results_dict

    @staticmethod
    def dict_to_obj_annot(result_dict: Dict[str, Any]) -> ObjectAnnotation:
        '''
        Converte um dicionário contendo bounding box e keypoints no formato Tiffany
        para um objeto `ObjectAnnotation` do protocolo IS.

        Parâmetros:
            result_dict (dict): Dicionário com a chave 'boxes'.

        Retorna:
            ObjectAnnotation: Anotação com rótulo, score e região.
        '''
        box = result_dict["boxes"][0]
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
        )