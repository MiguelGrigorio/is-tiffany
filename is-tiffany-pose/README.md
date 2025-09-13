# is-tiffany-pose

> Real-time pose estimation service for Tiffany, the hexapod robot, part of LabSEA's Intelligent Space ('is') ecosystem.

## Overview

**is-tiffany-pose** is a microservice designed to compute the 3D pose of **Tiffany**, a hexapod robot, using keypoints detected from multiple fixed cameras within the **Intelligent Space (is)** framework.

While Tiffany itself moves autonomously, this service aggregates keypoints from multiple camera feeds, calculates the robot's center position and orientation, and provides real-time pose information to other microservices in the system.

This project leverages:

- Multi-camera keypoint fusion
- Pose estimation using triangulation
- Angle smoothing for stable orientation
- RPC interfaces to start/stop detection and retrieve the latest pose
- Distributed tracing with Zipkin support
- Scalable deployment with Docker and Kubernetes

---

## Architecture

- Each camera pod publishes detected keypoints to the `is-wire` network.
- `is-tiffany-pose` consumes keypoints from all camera pods, calculates the robot’s 3D center and orientation, and exposes RPC endpoints for other services to consume the pose.
- Angle history smoothing ensures stable orientation values even when keypoints flicker.

---

## Features

- Computes 3D position and yaw orientation of Tiffany
- Smooths angles to reduce noise from keypoint detection
- Thread-safe handling of keypoints and pose data
- RPC interface to trigger detection and retrieve latest pose
- Supports distributed tracing with Zipkin
- Docker- and Kubernetes-ready deployment

---

## Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU (optional, only if keypoint detection needs acceleration)
- RabbitMQ or AMQP broker for `is-wire` messaging
- Zipkin server for tracing (optional)

Install dependencies:

```bash
pip install -r requirements.txt
```
## Usage
### Running Locally
Set environment variables:
```bash
export BROKER_URI="amqp://guest:guest@10.10.2.211:30000"
export ZIPKIN_URI="http://10.10.2.211:30200"
```

Run the service:
```bash
python src/main.py
```

### RPC Endpoints

`Tiffany.GetPose`: Returns the latest pose as a Pose protobuf.

`Tiffany.StartDetections`: Starts detection threads for a given duration (in minutes, FloatValue).

#### Example: Sending RPC Requests
```bash
python etc/example/send_request.py
```
This example:

1. Sends a request to start pose detection threads for all cameras for a given duration.

2. Waits briefly to ensure detections start running.

3. Requests the latest calculated pose.

4. Prints the pose and RPC status.

#### Expected Output:
```json
RPC Status: { code=StatusCode.OK, why='Detections started successfully' }
Pose: position {
  x: 1.234
  y: 2.345
  z: 0.123
}
orientation {
  yaw: 45.67
}

```

## Deployment
### Docker
Pull the Docker image:
```bash
docker pull miguelgrigorio27/is-tiffany-pose:latest
```
Or build your own Docker image:
```bash
docker build -t yourusername/is-tiffany-pose:latest -f etc/docker/Dockerfile .
```
Run the container:
```bash
docker run -e BROKER_URI=$BROKER_URI -e ZIPKIN_URI=$ZIPKIN_URI yourusername/is-tiffany-pose:latest
```

### Kubernetes
Apply the Kubernetes manifests in `etc/k8s`:
```bash
kubectl apply -f etc/k8s/ConfigMap.yaml
kubectl apply -f etc/k8s/Service.yaml
```

## Camera Calibration

The service uses calibration files stored in `src/calibrations/` (`calib_rt1.npz` to `calib_rt4.npz`).
These files contain camera intrinsics, extrinsics, and distortion parameters required for triangulation.

## About Tiffany and Intelligent Space ('is')

Tiffany is a hexapod robot designed for autonomous navigation and interaction within an intelligent environment.
The Intelligent Space (is) ecosystem integrates multiple robotic and sensor components, enabling distributed intelligence.

`is-tiffany-pose` serves as the pose perception module, providing other microservices with stable, real-time 3D pose information for control, navigation, and planning.
