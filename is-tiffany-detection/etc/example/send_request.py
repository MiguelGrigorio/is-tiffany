from is_wire.core import Channel, Message, Subscription
from google.protobuf.wrappers_pb2 import FloatValue
from is_msgs.image_pb2 import ObjectAnnotations
import socket
import time

# Create a channel and a subscription for replies
channel = Channel("amqp://guest:guest@10.10.2.211:30000")
subscription = Subscription(channel)

# Creating the message to start the stream with a FloatValue content (1.0 minutes)
request = Message(content = FloatValue(value = 1.0), reply_to=subscription)

# Publishing the request to start the stream
channel.publish(request, topic="Tiffany.Detection.1.StartStream")

# Waiting for the reply with a timeout of 5 seconds
try:
    reply = channel.consume(timeout = 5.0)
    print('RPC Status: ', reply.status)
except socket.timeout:
    print('No reply :(')


time.sleep(5.0) # Wait a bit to ensure the stream has started

# Creating the message to get the detections
request = Message(reply_to=subscription)
channel.publish(request, topic="Tiffany.Detection.1.GetDetection")

# Waiting for the reply with a timeout of 5 seconds
try:
    reply = channel.consume(timeout = 5.0)
    print('RPC Status: ', reply.status)
    print('Detection: ', reply.unpack(ObjectAnnotations))
except socket.timeout:
    print('No reply :(')