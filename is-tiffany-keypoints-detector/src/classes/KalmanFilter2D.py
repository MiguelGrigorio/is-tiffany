import numpy as np
import cv2

class KalmanFilter2D:
    def __init__(self, process_noise = 1e-6, measurement_noise = 1e-1):
        # Cria um filtro de Kalman com 4 variáveis de estado (x, y, vx, vy) e 2 medidas (x, y)
        self.kf = cv2.KalmanFilter(4, 2, 0)
        
        # Matriz de transição (modelo de velocidade constante)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)
        
        # Matriz de medida (só observamos x e y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)
        
        # Covariância do processo (ajuste para suavização)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Covariância da medida (ajuste para confiança nas medidas)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Covariância do erro (inicial)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Estado inicial (será atualizado na primeira medida)
        self.kf.statePost = np.zeros((4,1), dtype=np.float32)
    
    def update(self, x, y):
        # Faz a predição
        prediction = self.kf.predict()
        
        # Corrige com a nova medida
        measurement = np.array([[x], [y]], dtype=np.float32)
        estimated = self.kf.correct(measurement)
        
        # Retorna a posição suavizada (x, y)
        return estimated[0][0], estimated[1][0]

