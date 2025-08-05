from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.core import Subscription, Logger, AsyncTransport
from .StreamChannel import StreamChannel
import re

class Connection:
    def __init__(self, broker_uri: str, zipkin_uri: str, camera_id: int, service_name: str, log: Logger) -> None:
        self.channel_camera = StreamChannel(broker_uri)
        self.channel_detection = StreamChannel(broker_uri)
        
        Subscription(self.channel_camera).subscribe(f"CameraGateway.{camera_id}.Frame")
        Subscription(self.channel_detection).subscribe(f"Tiffany.{camera_id}.Detection")
        log.info(f"Connected to broker: {broker_uri} for camera ID: {camera_id}")
        
        self.exporter = self.create_exporter(service_name, zipkin_uri, log)
        log.info(f"Exporter created for service: {service_name} with Zipkin URI: {zipkin_uri}")
        
    @staticmethod
    def create_exporter(service_name: str, uri: str, log: Logger) -> ZipkinExporter:
        '''
        Cria e retorna um ZipkinExporter para rastreamento distribuído.

        Parâmetros:
            service_name (str): Nome do serviço que será exibido no Zipkin.
            uri (str): URI do Zipkin no formato 'http://<hostname>:<port>'.
            log (Logger): Logger para mensagens de erro ou debug.

        Retorna:
            ZipkinExporter: Objeto configurado para exportar traces para o Zipkin.

        Levanta:
            ValueError: Se a URI estiver em formato inválido.
        '''
        zipkin_ok = re.match(r"http://([a-zA-Z0-9\.-]+)(:(\d+))?", uri)
        if not zipkin_ok:
            log.critical('Invalid zipkin uri "{}", expected http://<hostname>:<port>', uri)
            raise ValueError(f"Invalid Zipkin URI: {uri}")

        exporter = ZipkinExporter(
            service_name = service_name,
            host_name = zipkin_ok.group(1),
            port = zipkin_ok.group(3),
            transport = AsyncTransport,
        )
        return exporter