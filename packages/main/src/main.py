#!/usr/bin/env python3

from copy import deepcopy
import os
import numpy as np
import math
import yaml
import cv2
import os
from tag import Tag
from dt_apriltags import Detector
from duckietown_msgs.msg import Twist2DStamped, LanePose, ButtonEvent, WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType
import rospy   
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import cpu_count

class Lab5(DTROS):
    def __init__(self, node_name):
        super(Lab5, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
     
        rospy.on_shutdown(self.on_shutdown)
        self.r = rospy.Rate(5)
        self.bridge = CvBridge()
        self.start = rospy.get_time()
        self.latest_gray = None
        vehicle_name = os.environ['VEHICLE_NAME'] if 'VEHICLE_NAME' in os.environ else 'csc22917'
        self.pub_action = rospy.Publisher("/{}/joy_mapper_node/car_cmd".format(vehicle_name), Twist2DStamped, queue_size=10)
        self.img_sub = rospy.Subscriber("/{}/camera_node/image/compressed".format(vehicle_name), CompressedImage, self.img_callback)
        self.__id_map = {
            0: [0, 0, 0.3048, 0, 0, 0],
            1: [0.3048, 0, 0.6096, 0, math.pi / 2, 0],
            2: [0.6096, 0, 0.3048, 0, math.pi, 0],
            3: [0.3048, 0, 0, 0, -math.pi / 2, 0],
            4: [0.3048, 0, 0, 0, -math.pi / 2, 0]
        }

        # TODO: add information about tags
        TAG_SIZE = 0.08
        FAMILIES = "tagStandard41h12"
        self.tags = Tag(TAG_SIZE, FAMILIES)

        # Add information about tag locations
        # Function Arguments are id, x, y, z, theta_x, theta_y, theta_z (euler) 
        # for example, self.tags.add_tag( ... 

        # Load camera parameters
        with open("/data/config/calibrations/camera_intrinsic/{}.yaml".format(vehicle_name)) as file:
            camera_list = yaml.load(file, Loader = yaml.FullLoader)

        self.camera_intrinsic_matrix = np.array(camera_list['camera_matrix']['data']).reshape(3, 3)
        self.distortion_coeff = np.array(camera_list['distortion_coefficients']['data']).reshape(5, 1)
        detector = Detector(families=FAMILIES, nthreads=cpu_count())

        while not rospy.is_shutdown():
            if self.latest_gray is None:
                rospy.loginfo_throttle(1., 'Waiting to grab camera image...')
                continue

            gray = deepcopy(self.latest_gray)
            detected_tags = self.detect(gray, detector=detector)
            rospy.logwarn_throttle(1., 'Detected AprilTags: {}'.format([x.tag_id for x in detected_tags]))
            if len(detected_tags) > 0:
                pose_estimates = []
                for tag in detected_tags:
                    tag_id = tag.tag_id
                    
                    pose_R = tag.pose_R
                    pose_t = tag.pose_t
                    
                    camera_rotation_x = 180 * np.arctan(pose_R[1][0] / pose_R[0][0]) / math.pi
                    camera_rotation_y = 180 * np.arcsin(-1 * pose_R[2][0]) / math.pi
                    camera_rotation_z = 180 * np.arctan(pose_R[2][1] / pose_R[2][2]) / math.pi
                    
                    R = self.tags.eulerAnglesToRotationMatrix(camera_rotation_x, camera_rotation_y, camera_rotation_z)
                    t = self.tags.TranslationVector(*pose_t).reshape(3, 1)
                    self.tags.add_tag(tag_id, *self.__id_map[tag_id])
                    estimate = self.tags.estimate_pose(tag_id, R, t)
                    pose_estimates.append(estimate)

                    # rospy.logerr_throttle(1., 'Pose estimate for : {}'.format(estimate))
                    
                    # P = np.array([
                    #     [1, 0, 0],
                    #     [0, -1, 0],
                    #     [0, 0, -1]
                    # ])
                    # unofficial_tag_position = P @ pose_R.T @ (-1 * pose_t)
                    # global_position = R_g @ unofficial_tag_position + t_g

                    # self.tags.add_tag(0, 0, 0, 0.3048, 0, 0, 0)
                    # self.tags.add_tag(1, 0.3048,0, 0.6096, 0, math.pi/2,0)
                    # self.tags.add_tag(2, 0.6096, 0, 0.3048, 0, math.pi, 0)
                    # self.tags.add_tag(3, 0.3048, 0, 0, 0, -math.pi/2, 0)

                    # print(type(pose_R))
                    # print(pose_R)

                    # camera_rotation_x = 180 * np.arctan(pose_R[1][0] / pose_R[0][0]) / math.pi
                    # camera_rotation_y = 180 * np.arcsin(-1 * pose_R[2][0]) / math.pi
                    # camera_rotation_z = 180 * np.arctan(pose_R[2][1] / pose_R[2][2]) / math.pi
                    # camera_x, camera_y, camera_z = pose_t

                    # print("rotation x = " + str(camera_rotation_x))
                    # print("rotation y = " + str(camera_rotation_y))
                    # print("rotation z = " + str(camera_rotation_z))
                    # print("X: " + str(camera_x))
                    # print("Y: " + str(camera_y))
                    # print("Z: " + str(camera_z))

                rospy.logerr_throttle(1., 'Pose estimate for : {}'.format(np.mean(pose_estimates, axis=0)))

            self.r.sleep()
        
        print('Doning')

    def img_callback(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        undistorted_img = self.undistort(img)
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        self.latest_gray = gray

    '''
        Takes a fisheye-distorted image and undistorts it

        Adapted from: https://github.com/asvath/SLAMDuck
    '''
    def undistort(self, img):
        height = img.shape[0]
        width = img.shape[1]

        newmatrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_intrinsic_matrix,
            self.distortion_coeff, 
            (width, height),
            1, 
            (width, height)
        )

        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_intrinsic_matrix, 
            self.distortion_coeff,  
            np.eye(3), 
            newmatrix, 
            (width, height), 
            cv2.CV_16SC2
        )

        undistorted_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
       
        return undistorted_image   

    '''
        Takes an images and detects AprilTags
    '''
    def detect(self, img, detector=None):
        PARAMS = [
            self.camera_intrinsic_matrix[0, 0],
            self.camera_intrinsic_matrix[1, 1],
            self.camera_intrinsic_matrix[0, 2],
            self.camera_intrinsic_matrix[1, 2]
        ]

        TAG_SIZE = 0.08
        
        if detector is None:
            detector = Detector(families="tagStandard41h12", nthreads=cpu_count())
        
        detected_tags = detector.detect(
            img, 
            estimate_tag_pose=True, 
            camera_params=PARAMS, 
            tag_size=TAG_SIZE
        )

        return detected_tags

    def on_shutdown(self):
        rospy.loginfo('Shutting down node...')
        self.stop()

    def stop(self, manual=True):
        rospy.loginfo('Stopping...')
        self.pub_action.publish(Twist2DStamped())
        if manual:
            rospy.signal_shutdown('Manual shutdown')

lab_5 = Lab5('lab_5_main')
