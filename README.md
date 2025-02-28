# Autonomous Lane Following and Object Detection Robot

## Overview
This project implements an autonomous lane-following robot with object detection using a Raspberry Pi and a camera module. The system uses OpenCV for lane detection, YOLO for object detection, and GPIOZero for motor control. The goal is to navigate a robot within lane boundaries while avoiding obstacles detected in its path.

## Features
- **Lane Detection**: Uses edge detection and Hough Transform to identify lane markings.
- **Object Detection**: Utilizes YOLO (You Only Look Once) to identify obstacles in the path.
- **Motor Control**: Dynamically adjusts motor speeds for steering and stopping.
- **Emergency Stop**: Stops the robot if an object is detected between lanes.

## Hardware Requirements
- Raspberry Pi 5
- Raspberry Pi Camera Module
- Motor Driver (L298N or equivalent)
- Two DC Motors
- DSI LCD Touchscreen (Optional for display output)
- Power Supply (Battery pack)

## Software Requirements
- Raspberry Pi OS (Bookworm or later)
- Python 3
- OpenCV
- GPIOZero
- Picamera2
- NumPy
- UltraLytics YOLO
