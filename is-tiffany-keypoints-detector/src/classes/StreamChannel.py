import socket
from is_wire.core import Channel
from is_wire.core import Message
from typing import Tuple

class StreamChannel(Channel):
    """
    Classe especializada para consumir apenas a última mensagem disponível de um canal IS.
    
    Herda de `is_wire.core.Channel` e sobrescreve o método `consume_last` para descartar
    mensagens antigas, retornando somente a mais recente.
    """

    def __init__(self, uri: str = "amqp://guest:guest@10.10.2.211:30000", exchange: str = "is"):
        """
        Inicializa o canal com URI e exchange padrão.

        Parâmetros:
            uri (str): URI do servidor AMQP.
            exchange (str): Nome da exchange a ser usada.
        """
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self, return_dropped: bool = False) -> Tuple[Message, int]:
        """
        Consome apenas a última mensagem disponível no canal, descartando as anteriores.

        Parâmetros:
            return_dropped (bool): Se True, retorna também o número de mensagens descartadas.

        Retorna:
            - Message: a última mensagem disponível.
            - (Message, int): se `return_dropped` for True, inclui o número de mensagens descartadas.
            - False: se nenhum dado estiver disponível no momento.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped) if return_dropped else msg
