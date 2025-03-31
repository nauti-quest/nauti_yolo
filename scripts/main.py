#! /usr/bin/env python3
import rospy
from YOLODetector.yolo_detector import YOLODetector

if __name__ == '__main__':
    rospy.init_node('YOLO Detector')
    rospy.loginfo("YOLO Detector running!")
    try:
        detector = YOLODetector()
    except Exception as ex:
        rospy.logwarn(rospy.get_name() + " : " + str(ex))
    rospy.spin()
