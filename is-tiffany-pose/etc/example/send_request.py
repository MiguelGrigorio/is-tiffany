from is_wire.core import Channel, Message, Subscription
from google.protobuf.wrappers_pb2 import FloatValue
from is_msgs.common_pb2 import Pose
import socket
import time

# Create a channel and a subscription for replies
channel = Channel("amqp://guest:guest@10.10.2.211:30000")
subscription = Subscription(channel)

# Start detection threads for 5 minutes
request = Message(content=FloatValue(value=5.0), reply_to=subscription)
channel.publish(request, topic="Tiffany.StartDetections")

# Wait for reply with timeout
try:
    reply = channel.consume(timeout=5.0)
    print('RPC Status: ', reply.status)
except socket.timeout:
    print('No reply :(')

# Wait a little to let detections run
time.sleep(5.0)

# Request the latest pose
request = Message(reply_to=subscription)
channel.publish(request, topic="Tiffany.GetPose")

try:
    reply = channel.consume(timeout=5.0)
    print('RPC Status: ', reply.status)
    pose: Pose = reply.unpack(Pose)
    print('Pose:', pose)
except socket.timeout:
    print('No reply :(')
