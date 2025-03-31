#! /usr/bin/env python3

import rospy
import rospkg
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from yolo_bbox.msg import BoundingBox
from ultralytics import YOLO
import ros_numpy
import time
import os

class YOLODetector():

    def __init__(self):
        input_topic = rospy.get_param(rospy.get_name() + "/input_topic", "/image")
        frame_rate = rospy.get_param(rospy.get_name() + "/frame_rate", "/rate")
        output_topic = rospy.get_param(rospy.get_name() + "/output_topic", "/detections")
        self.publish_image_output = rospy.get_param(rospy.get_name() + "/publish_image_output", True)
        output_image_topic = rospy.get_param(rospy.get_name() + "/output_image_topic", "/yolo_output")
        self.verbose = rospy.get_param(rospy.get_name() + "/verbose", True)

        # Load Model
        model_path = os.path.join(rospkg.RosPack().get_path('yolo_bbox'), 'model', 'yolov10b.pt')
        self.model = YOLO(model_path)

        # Create publishers / subscribers
        self.enable_sub = rospy.Subscriber('enable', Bool, self.enable_callback)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.detections_pub = rospy.Publisher(output_topic, BoundingBox, queue_size=100)
        self.image_pub = rospy.Publisher(output_image_topic, Image, queue_size=100)

        # Enable module
        self.enable = True
        
        # Rate control 
        self.last_detection_stamp = time.time()
        self.detection_interval = 1.0/frame_rate

    
    def image_callback(self, msg):
        # Check if module enabled
        if not self.enable:
            return
        
        # Control detection frame rate
        if (self.detection_interval > (time.time() - self.last_detection_stamp)):
            return

        self.last_detection_stamp = time.time()
        image = ros_numpy.numpify(msg)
        result = self.model(image, verbose=self.verbose)[0]

        for x in range (0, len(result.boxes.cls)):
            detection_msg = BoundingBox()
            detection_msg.class_name = result.names[int(result.boxes.cls[x])]
            detection_msg.top_left_x = int(result.boxes.xywh[0][0])
            detection_msg.top_left_y = int(result.boxes.xywh[0][1])
            detection_msg.width = int(result.boxes.xywh[0][2])
            detection_msg.height = int(result.boxes.xywh[0][3])
            detection_msg.class_prob = float(result.boxes.conf[x])
            detection_msg.image_width = msg.width
            detection_msg.image_height = msg.height
            
            detection_msg.top_left_x = int(result.boxes.xywh[0][0] - detection_msg.width/2)
            detection_msg.top_left_y = int(result.boxes.xywh[0][1] - detection_msg.height/2)
            self.detections_pub.publish(detection_msg)
            print(detection_msg)

        if (self.publish_image_output):
            result_image = result.plot(show=False)
            self.image_pub.publish(ros_numpy.msgify(Image, result_image, encoding="rgb8"))
        
        # Check execution time
        if (self.detection_interval < (time.time() - self.last_detection_stamp)):
            rospy.logwarn("%s : Execution took %f seconds. Max time interval %f seconds. Reduce frame rate!"%(
                rospy.get_name(),(time.time()-self.last_detection_stamp), self.detection_interval))

    

    def enable_callback(self, msg):
        self.enable = msg.data
        if self.enable:
            rospy.loginfo(rospy.get_name() + " Enabled")
        else:
            rospy.loginfo(rospy.get_name() + "Disabled")


if __name__ == '__main__':
    rospy.init_node('yolo_detector')
    rospy.loginfo("YOLO Detector Started")
    try:
        detector = YOLODetector()
    except Exception as ex:
        rospy.logwarn("YOLO Detector error" + str(ex))

    rospy.spin()

