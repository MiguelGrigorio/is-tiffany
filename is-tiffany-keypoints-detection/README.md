# is-tiffany-keypoints-detection

> Real-time keypoints detection service for Tiffany, the hexapod robot, part of LabSEA's Intelligent Space ('is') ecosystem.

## Overview

**is-tiffany-keypoints-detection** is a microservice designed to process video feeds from external cameras (e.g., CCTV) and provide real-time **keypoints detection** results for **Tiffany**, a hexapod robot operating within the **Intelligent Space (is)** framework.

While Tiffany is a mobile hexapod robot, this service processes images streamed from fixed cameras deployed in the environment, enabling Tiffany to determine its **pose and orientation** accurately in real time.

This project combines PyTorch-based YOLO models for **pose estimation** with the `is-wire` messaging middleware to provide:

- Continuous keypoints detection from camera feeds
- RPC interfaces to start/stop detection and streaming
- Distributed tracing with Zipkin support
- Scalable deployment with Docker and Kubernetes

---

## Architecture

- The camera gateway captures images from fixed CCTV cameras and publishes them on an `is-wire` stream.
- The keypoints detection service consumes these images, runs pose estimation models, and publishes annotated keypoints results.

---

## Features

- Real-time consumption of camera images
- Thread-safe detection and streaming management
- RPC interface to control detection and stream durations dynamically
- Distributed tracing support via Zipkin
- Easily deployable with Docker and Kubernetes manifests

---

## Installation

### Requirements

- Python 3.9+
- RabbitMQ or AMQP broker for `is-wire` messaging
- Zipkin server for tracing (optional)

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- CUDA-compatible GPU
- RabbitMQ or AMQP broker for `is-wire` messaging
- Zipkin server for tracing (optional)

Install dependencies with:

```bash
pip install -r requirements.txt
```
## Usage
### Running Locally
Set environment variables:
```bash
export BROKER_URI="amqp://guest:guest@10.10.2.211:30000"
export ZIPKIN_URI="http://10.10.2.211:30200"
export CAMERA_ID=1
```

Run the service:
```bash
python src/main.py
```
### RPC Endpoints
`Tiffany.Keypoints.{camera_id}.GetDetection`
Returns the latest keypoints detected by the specified camera as an `ObjectAnnotations` protobuf.

`Tiffany.Keypoints.{camera_id}.StartStream`
Starts streaming keypoints from the specified camera for a given duration (in minutes, `FloatValue`). Returns a `Status` message indicating success or failure.

`Tiffany.Keypoints.{camera_id}.StartDetection`
Starts continuous keypoints detection on the specified camera for a given duration (in minutes, `FloatValue`). Returns a `Status` message indicating success or failure.

#### Example: Sending RPC Requests
Use the example script to start a stream and fetch detections:
```bash
python etc/example/send_request.py
```
This script:

1. Sends a request to start a keypoints detection stream for a defined duration.

2. Waits for the stream to initialize.

3. Requests the latest keypoints detection result.

4. Prints the keypoints coordinates and detection confidence.

#### Expected Output:
```json
RPC Status: { code=StatusCode.OK why='Stream started' }

RPC Status: { code=StatusCode.OK why='' }
Detection: objects {
  label: "Tiffany"
  score: 0.79606918
  region {
    vertices {
      x: 157
      y: 593
    }
    vertices {
      x: 184
      y: 625
    }
  }
  keypoints {
    id: 0
    score: 0.87
    position {
      x: 170
      y: 600
    }
  }
  keypoints {
    id: 1
    score: 0.85
    position {
      x: 180
      y: 610
    }
  }
}
resolution {
  height: 720
  width: 1280
}
frame_id: 1
```

## Deployment
### Docker
Pull the Docker image:
```bash
docker pull miguelgrigorio27/is-tiffany-keypoints-detection:latest
```
Or build your own Docker image:
```bash
docker build -t yourusername/is-tiffany-keypoints-detection:latest -f etc/docker/Dockerfile .
```
Run the container:
```bash
docker run -e BROKER_URI=$BROKER_URI -e ZIPKIN_URI=$ZIPKIN_URI -e CAMERA_ID=$CAMERA_ID yourusername/is-tiffany-keypoints-detection:latest
```

### Kubernetes
Apply the Kubernetes manifests in `etc/k8s`:
```bash
kubectl apply -f etc/k8s/ConfigMap.yaml
kubectl apply -f etc/k8s/Service.yaml
```
The StatefulSet deploys 4 replicas, each assigned a unique camera ID from the pod hostname ordinal index.


## About Tiffany and Intelligent Space ('is')

Tiffany is a hexapod robot designed for autonomous navigation and interaction within an intelligent environment.

The Intelligent Space (is) ecosystem provides middleware and infrastructure to integrate multiple robotic and sensor components, enabling a distributed intelligent system.

The `is-tiffany-keypoints-detection` service fits into this ecosystem as the pose perception module, interpreting environmental camera data to determine Tiffany's keypoints for downstream navigation and interaction.