# is-tiffany

> Distributed microservices ecosystem for [Tiffany](https://github.com/Penguin-Lab/tiffany), the hexapod robot, operating within [LabSEA](https://github.com/Lab-SEA)'s Intelligent Space ('is') framework.

This repository contains three main services:

**is-tiffany-detection**: Real-time tiffany detection from camera feeds.

**is-tiffany-keypoints-detection**: Real-time tiffany keypoints detection from camera feeds to estimate Tiffany’s pose and orientation.

**is-tiffany-pose**: Computes Tiffany’s 3D position and orientation by fusing keypoints from multiple cameras.

Each service runs independently as a microservice, exposing RPC endpoints via the `is-wire` messaging middleware, and can be deployed using Docker and Kubernetes.

## Core Concepts

**Microservices Architecture**: Each component (detection, keypoints, pose) runs independently and communicates via RPC using `is-wire`.

**RPC Endpoints**: Services expose topics like `Tiffany.Detection.{camera_id}.GetDetection` or `Tiffany.GetPose` to retrieve information on demand.

**Docker & Kubernetes**: Each service can run locally in Docker or be orchestrated in Kubernetes, optionally with multiple camera pods.

**Distributed Tracing**: Zipkin integration allows monitoring service calls and performance metrics across the microservices.

**Real-time Perception**: The detection and keypoints services consume live camera feeds and provide annotated results for downstream pose computation.

**Pose Estimation**: The pose service fuses keypoints from multiple cameras and applies triangulation and angle smoothing to produce stable 3D pose estimates.

## About Tiffany and Intelligent Space ('is')

[Tiffany](https://github.com/Penguin-Lab/tiffany) is a hexapod robot designed for autonomous navigation and interaction in a sensor-rich environment.
The Intelligent Space (is) ecosystem provides the middleware, messaging, and infrastructure needed to integrate multiple robotic and sensor components, enabling distributed intelligence.

The `is-tiffany` services collectively allow Tiffany to perceive its surroundings, estimate its pose, and navigate autonomously, forming a complete perception and control stack.