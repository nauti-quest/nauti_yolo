#! /usr/bin/env python3

import rospy
import rospkg
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image
from yolo_bbox.msg import BoundingBox
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

class YOLODetector():

    def __init__(self):
        input_topic = rospy.get_param(rospy.get_name() + "/input_topic", "/image")
        frame_rate = rospy.get_param(rospy.get_name() + "/frame_rate", 10)
        output_topic = rospy.get_param(rospy.get_name() + "/output_topic", "/detections")
        self.publish_image_output = rospy.get_param(rospy.get_name() + "/publish_image_output", True)
        output_image_topic = rospy.get_param(rospy.get_name() + "/output_image_topic", "/yolo_output")
        self.verbose = rospy.get_param(rospy.get_name() + "/verbose", True)

        # Load Model
        model_path = os.path.join(rospkg.RosPack().get_path('yolo_bbox'), 'model', 'yolov10n.pt')
        self.model = YOLO(model_path)

        # Create publishers / subscribers
        self.enable_sub = rospy.Subscriber('enable', Bool, self.enable_callback)
        self.image_sub = rospy.Subscriber(input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.detections_pub = rospy.Publisher(output_topic, BoundingBox, queue_size=100)
        self.image_pub = rospy.Publisher(output_image_topic, Image, queue_size=100)

        # Enable module
        self.enable = True
        
        # Rate control 
        self.last_detection_stamp = time.time()
        self.detection_interval = 1.0 / frame_rate

    
    def image_callback(self, msg):
        if not self.enable:
            return
        
        if (self.detection_interval > (time.time() - self.last_detection_stamp)):
            return

        self.last_detection_stamp = time.time()

        # Convert compressed image to OpenCV image (NumPy array)
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            rospy.logwarn("Failed to decode compressed image.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model(image_rgb, verbose=self.verbose)[0]

        height, width, _ = image.shape

        for x in range(0, len(result.boxes.cls)):
            detection_msg = BoundingBox()
            detection_msg.class_name = result.names[int(result.boxes.cls[x])]
            detection_msg.width = int(result.boxes.xywh[x][2])
            detection_msg.height = int(result.boxes.xywh[x][3])
            detection_msg.top_left_x = int(result.boxes.xywh[x][0] - detection_msg.width / 2)
            detection_msg.top_left_y = int(result.boxes.xywh[x][1] - detection_msg.height / 2)
            detection_msg.class_prob = float(result.boxes.conf[x])
            detection_msg.image_width = width
            detection_msg.image_height = height
            self.detections_pub.publish(detection_msg)
            #print(detection_msg)

        if self.publish_image_output:
            # Instead of using result.plot() which draws labels automatically
            # Create a custom visualization
            result_image = image.copy()  # Use the original image as base
            
            # Draw each box manually
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence score only (no label)
                conf_text = f"{conf:.2f}"
                cv2.putText(result_image, conf_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to ROS image message
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            image_msg = Image()
            image_msg.header = msg.header
            image_msg.height = height
            image_msg.width = width
            image_msg.encoding = "rgb8"
            image_msg.is_bigendian = 0
            image_msg.step = 3 * width
            image_msg.data = result_image_rgb.tobytes()
            self.image_pub.publish(image_msg)

        if (self.detection_interval < (time.time() - self.last_detection_stamp)):
            rospy.logwarn("%s : Execution took %f seconds. Max time interval %f seconds. Reduce frame rate!" % (
                rospy.get_name(), (time.time() - self.last_detection_stamp), self.detection_interval))

    def enable_callback(self, msg):
        self.enable = msg.data
        if self.enable:
            rospy.loginfo(rospy.get_name() + " Enabled")
        else:
            rospy.loginfo(rospy.get_name() + " Disabled")

if __name__ == '__main__':
    rospy.init_node('yolo_detector')
    rospy.loginfo("YOLO Detector Started")
    try:
        detector = YOLODetector()
    except Exception as ex:
        rospy.logwarn("YOLO Detector error: " + str(ex))

    rospy.spin()