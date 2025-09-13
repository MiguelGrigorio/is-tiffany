from is_wire.rpc import ServiceProvider, LogInterceptor, TracingInterceptor
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.core import Logger, AsyncTransport
from .StreamChannel import StreamChannel
import re


class Connection:
    """Manages connections with the message broker and the Zipkin tracing server.

    This class encapsulates the logic to establish communication channels,
    provide RPC services, and configure distributed tracing with Zipkin.

    Attributes:
        log (Logger): An instance of a logger for application logs.
        provider (ServiceProvider): Manages and exposes the RPC services.
        exporter (ZipkinExporter): Exporter for sending traces to Zipkin.
        broker_uri (str): URI of the message broker.
        zipkin_uri (str): URI of the Zipkin server.
        camera_id (int): ID of the camera to subscribe to.
        service_name (str): The name of this service.
    """

    def __init__(
        self,
        broker_uri: str,
        zipkin_uri: str,
        camera_id: int,
        service_name: str
    ) -> None:
        """Initializes the connection manager.

        Args:
            broker_uri (str): The URI of the message broker (e.g., "amqp://guest:guest@localhost:5672").
            zipkin_uri (str): The URI of the Zipkin tracing server (e.g., "http://localhost:9411").
            camera_id (int): The unique identifier of the camera feed to subscribe to.
            service_name (str): The name of this service, used for identification in Zipkin.
        """
        self.provider = ServiceProvider(StreamChannel(broker_uri))
        log = LogInterceptor()
        self.provider.add_interceptor(log)
        self.log = log.log

        self.log.info(f"Successfully connected to broker at {broker_uri} for camera ID {camera_id}")
        
        self.exporter = self.create_exporter(service_name, zipkin_uri, self.log)
        self.provider.add_interceptor(TracingInterceptor(self.exporter))
        self.log.info(f"Zipkin exporter initialized for service '{service_name}' with URI: {zipkin_uri}")

        self.broker_uri = broker_uri
        self.zipkin_uri = zipkin_uri
        self.camera_id = camera_id
        self.service_name = service_name

    @staticmethod
    def create_exporter(service_name: str, uri: str, log: Logger) -> ZipkinExporter:
        """Creates and configures a ZipkinExporter for distributed tracing.

        This method validates the Zipkin URI format and initializes an exporter
        with the provided service name and an asynchronous transport layer.

        Args:
            service_name (str): Name of the service to be displayed in Zipkin.
            uri (str): URI of the Zipkin server, expected in the format 'http://<hostname>:<port>'.
            log (Logger): An instance of a logger to record messages, especially errors.

        Returns:
            ZipkinExporter: A configured instance of the exporter, ready to send traces.

        Raises:
            ValueError: If the provided URI does not match the expected format.
        """
        zipkin_ok = re.match(r"http://([a-zA-Z0-9\.-]+)(:(\d+))?", uri)
        if not zipkin_ok:
            log.critical(f"Invalid Zipkin URI '{uri}'. " 
                          "Expected format: http://<hostname>:<port>")
            raise ValueError(f"Invalid Zipkin URI: {uri}")

        exporter = ZipkinExporter(
            service_name=service_name,
            host_name=zipkin_ok.group(1),
            port=int(zipkin_ok.group(3)) if zipkin_ok.group(3) else 9411,  # Default port 9411
            transport=AsyncTransport,
        )
        return exporter
