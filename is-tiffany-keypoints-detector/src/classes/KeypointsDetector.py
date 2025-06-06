from is_msgs.image_pb2 import ObjectAnnotation, BoundingPoly, Vertex, PointAnnotation
from ultralytics.engine.results import Results
from .KalmanFilter2D import KalmanFilter2D
from typing import List, Dict, Any
from is_wire.core import Logger
from ultralytics import YOLO
import numpy as np

class KeypointsDetector():
    def __init__(self, model_path: str, logger: Logger, device: str = "cuda") -> None:
        self.model: YOLO = YOLO(model_path)
        self.model.to(device)
        logger.info(f"Loaded model: {model_path}")
        self.kp_0 = KalmanFilter2D(process_noise=1e-5, measurement_noise=10**(-2.5))
        self.kp_1 = KalmanFilter2D(process_noise=1e-5, measurement_noise=10**(-2.5))

    def predict(self, img: np.ndarray) -> Results:
        results = self.model.predict(source=img, imgsz=96, verbose = False)
        return results[0]
    
    def results_to_dict(self, results: Results, offset: np.ndarray) -> Dict[str, List[dict]]:
        '''
        Converte os resultados de detecção e pose do modelo YOLOv8 para um dicionário.

        Parâmetros:
            results (Results): Objeto de resultados da detecção do Ultralytics YOLOv8.
            offset (np.ndarray): Vetor 2D representando o deslocamento da região de interesse (ROI)
                            a ser somado aos keypoints (ex: np.array([x_offset, y_offset])).

        Retorna:
            dict: Um dicionário com duas chaves:
                - "boxes": lista de dicionários contendo a confiança ("conf") e coordenadas "xyxy".
                - "keypoints": lista de keypoints, cada um com sua "conf" e coordenadas ajustadas "xy".
        '''
        results_dict = {
            "boxes": [],
            "keypoints": []
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
            k0_raw = (results_dict["keypoints"][0]["xy"][0], results_dict["keypoints"][0]["xy"][1])
            k1_raw = (results_dict["keypoints"][1]["xy"][0], results_dict["keypoints"][1]["xy"][1])
            k0_smooth = self.kp_0.update(k0_raw[0], k0_raw[1])
            k1_smooth = self.kp_1.update(k1_raw[0], k1_raw[1])
            results_dict["keypoints"][0]["xy"][0] = k0_smooth[0]
            results_dict["keypoints"][0]["xy"][1]= k0_smooth[1]
            results_dict["keypoints"][1]["xy"][0] = k1_smooth[0]
            results_dict["keypoints"][1]["xy"][1] = k1_smooth[1]
        return results_dict

    @staticmethod
    def dict_to_obj_annot(result_dict: Dict[str, Any]) -> ObjectAnnotation:
        '''
        Converte um dicionário contendo bounding box e keypoints no formato Tiffany
        para um objeto `ObjectAnnotation` do protocolo IS.

        Parâmetros:
            result_dict (dict): Dicionário com as chaves 'boxes' e 'keypoints'.

        Retorna:
            ObjectAnnotation: Anotação com rótulo, score, região e keypoints.
        '''
        box = result_dict["boxes"][0]
        kps = result_dict["keypoints"]
        # Normalizando a confiança da terceira casa decimal
        a = 100*(kps[0]["conf"]-0.99)
        b = 100*(kps[1]["conf"]-0.99)
        
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
                    id=0,
                    score=a,
                    position=Vertex(x=kps[0]["xy"][0], y=kps[0]["xy"][1]),
                ),
                PointAnnotation(
                    id=1,
                    score=b,
                    position=Vertex(x=kps[1]["xy"][0], y=kps[1]["xy"][1]),
                ),
            ],
        )