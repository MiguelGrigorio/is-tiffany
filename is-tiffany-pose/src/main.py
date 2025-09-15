from google.protobuf.wrappers_pb2 import FloatValue
from google.protobuf.empty_pb2 import Empty
from classes import Connection, Threading
from is_msgs.common_pb2 import Pose
from is_wire.core import Status
import numpy as np
import os


def main() -> None:
    broker_uri = os.environ.get("broker_uri", "amqp://guest:guest@10.10.2.211:30000")
    zipkin_uri = os.environ.get("zipkin_uri", "http://10.10.2.211:30200")

    service_name = f"Tiffany.Pose"

    c = Connection(broker_uri, zipkin_uri, service_name)
    provider = c.provider
    
    parameters = {
        1: dict(np.load(f'calibrations/calib_rt1.npz')),
        2: dict(np.load(f'calibrations/calib_rt2.npz')),
        3: dict(np.load(f'calibrations/calib_rt3.npz')),
        4: dict(np.load(f'calibrations/calib_rt4.npz'))
    }
    threading_instance = Threading(c, parameters)
    provider.delegate(
        topic = f"Tiffany.GetPose",
        function = threading_instance.get_last_pose,
        request_type = Empty,
        reply_type = Pose
    )
    provider.delegate(
        topic= f"Tiffany.StartDetections",
        function=threading_instance.start_detections,
        request_type=FloatValue,
        reply_type=Status
    )
    provider.run()
    

if __name__ == "__main__":
    main()
