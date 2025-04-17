# nauti-yolo-ros

A ROS package that integrates YOLOv10 object detection with ROS. This package processes camera feed and publishes bounding box information that can be used by state machine nodes for further processing.

## Overview

This package provides object detection capabilities using YOLOv10 and publishes the results as ROS messages. It supports:
- Object detection using YOLOv10
- Publishing bounding box information
- Visualization of detection results

## Dependencies

- ROS (Robot Operating System)
- Python 3.8+
- YOLOv10
- OpenCV
- Additional ROS dependencies (defined in package.xml)

## Installation

1. Clone this package into your catkin workspace's src directory:
```bash
cd ~/catkin_ws/src
git clone [repository-url]
```

2. Build the workspace using catkin:
```bash
cd ~/catkin_ws
catkin build
```

3. Source your workspace:
```bash
source ~/catkin_ws/devel/setup.bash
```

## Configuration

Before running the node, configure the camera input source:

1. Navigate to the config folder
2. Open `parameters.yaml`
3. Adjust the `input_topic` parameter to match your camera's ROS topic

## Usage

Launch the YOLO detector node using:

```bash
roslaunch yolo_bbox yolo_detector.launch
```

## Topics

### Published Topics
- `/detections` - Publishes BoundingBox messages containing detection information
- `/yolo_output` - Publishes the processed image with visualized bounding boxes

### Subscribed Topics
- Camera input topic (configurable in parameters.yaml)

## Visualization

To view the detection results:
1. Start rqt_image_view
2. Select the `/yolo_output` topic to see the processed image with bounding boxes

## Message Types

This package uses custom message types:
- `BoundingBox.msg` - Contains information about individual detections

## Support
For issues and questions, please open an issue in this repository.

