#!/usr/bin/env python

##
# Extended and Customized by Giannis(Ioannis) Petousakis, Extreme Robotics Lab, University of Birmingham 2021
# Massimiliano Patacchiola 2020
#
# The MIT License (MIT)
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import rospy
import cv2
import numpy as np
import os
import tensorflow as tf

from deepgaze.face_detection import HaarFaceDetector
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

# Import of variable type for ROS publisher
from std_msgs.msg import Float32


# from operator import itemgetter


def Exponential_smoothing_filter(face_counter, exponential_smoothing_threshold, smoothing_factor, angle, angle_sum,
                                 angle_avg):
    if (face_counter <= exponential_smoothing_threshold):
        angle_sum += angle
        angle_avg = angle_sum / face_counter
        print(abs(angle_avg))

    elif (face_counter > exponential_smoothing_threshold):
        angle_avg = smoothing_factor * angle + (1 - smoothing_factor) * angle_avg
        print (abs(angle_avg), abs(angle))

    return angle_sum, angle_avg


def yaw2rotmat(yaw):
    x = 0.0
    y = 0.0
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3, 3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh * sb - ch * sa * cb
    rot[0][2] = ch * sa * sb + sh * cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh * sa * cb + ch * sb
    rot[2][2] = -sh * sa * sb + ch * cb
    return rot

def main():


    rospy.init_node('headang_pub', anonymous=False)  # ROS node init

    CPublisher = rospy.Publisher('cognitive_availability', Float32, queue_size=10)


    #rate = rospy.Rate(15) // disabled rate in order to have maximum refresh rate on ROS

    while not rospy.is_shutdown():

        # Defining the video capture object

        video_capture = cv2.VideoCapture(0)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter("./original.avi", fourcc, 24.0, (,2160))

        if (video_capture.isOpened() == False):
            print("Error: the resource is busy or unvailable")
        else:
            print("The video source has been opened correctly...")

        # Obtaining the CAM dimension
        cam_w = int(video_capture.get(3))
        cam_h = int(video_capture.get(4))

        print (cam_w, cam_h)

        c_x = cam_w / 2
        c_y = cam_h / 2
        # c_x = 640/2
        # c_y = 480/2

        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x

        camera_matrix = np.float32([[f_x, 0.0, c_x],
                                    [0.0, f_y, c_y],
                                    [0.0, 0.0, 1.0]])

        # print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
        # Distortion coefficients
        camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

        # Initialisation

        no_face_counter = 0

        # Exponential Smoothing Filter initialisation

        face_counter = 0
        exp_sm_thres = 25
        sf = 0.03  # Adjust smoothing factor to set the window frame for the calculation of the average
        yaw_sum = 0
        yaw_avg = 0
        pitch_avg = 0
        pitch_sum = 0
        roll_sum = 0
        roll_avg = 0
        CognAv_avg = 0
        CognAv_sum = 0
        pitch_avg_cnn = 0
        pitch_sum_cnn = 0
        yaw_avg_cnn = 0
        yaw_sum_cnn = 0

        now = 0

        # Variables that identify the face
        # position in the main frame.
        face_x1 = 0
        face_y1 = 0
        face_x2 = 0
        face_y2 = 0
        face_w = 0
        face_h = 0

        # Variables that identify the ROI
        # position in the main frame.
        roi_x1 = 0
        roi_y1 = 0
        roi_x2 = cam_w
        roi_y2 = cam_h
        roi_w = cam_w
        roi_h = cam_h
        roi_resize_w = int(cam_w / 10)
        roi_resize_h = int(cam_h / 10)

        hfd = HaarFaceDetector("../../etc/xml/haarcascade_frontalface_alt.xml", "../../etc/xml/haarcascade_profileface.xml")

        sess = tf.Session()  # Launch the graph in a session.
        my_head_pose_estimator = CnnHeadPoseEstimator(sess)  # Head pose estimation object

        my_head_pose_estimator.load_yaw_variables(os.path.realpath("../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))
        my_head_pose_estimator.load_roll_variables(os.path.realpath("../../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
        my_head_pose_estimator.load_pitch_variables(os.path.realpath("../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))

        while (True):

            ret, frame = video_capture.read()
            if (frame is None): break  # check for empty frames (en of video)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face = hfd.returnFacePosition(frame_gray, runFrontal=True, runFrontalRotated=True,
                                          runLeft=True, runRight=True,
                                          frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2,
                                          leftScaleFactor=1.15, rightScaleFactor=1.15,
                                          minSizeX=64, minSizeY=64,
                                          rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=0)
            # print (face)
            face_x1 = int(face[0])
            face_y1 = int(face[1])
            face_x2 = int(face_x1 + face[2])
            face_y2 = int(face_y1 + face[3])
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), [255, 0, 0])

            #Counter for measuring the lost frames and restarting the head capture process in case we have constant fail
            # !! have to verify it is needed along with the unused parameters.

            if (face[4] == 0):
                no_face_counter += 1

            if (face[4] == 50):
                face[4] = 0
                roi_x1 = 0    #probaly remove box resizing parameters roi x1 etc
                roi_y1 = 0
                roi_x2 = cam_w
                roi_y2 = cam_h
                roi_w = cam_w
                roi_h = cam_h

                # ESM Value reset on loss of Head
                face_counter = 0
                yaw_avg = 0
                yaw_sum = 0
                yaw_avg_cnn = 0
                yaw_sum_cnn = 0
                yaw = 0
                pitch_avg = 0
                pitch_sum = 0
                pitch = 0
                roll_sum = 0
                roll_avg = 0
                pitch_avg_cnn = 0
                pitch_sum_cnn = 0
                roll = 0
                CognAv = 0
                CognAv_sum = 0
                CognAv_avg = 0



            # If face is detected then we crop resize and feed the face image into the head angle estimator


            if (face[4] > 0):
                no_face_counter = 0
                face_counter += 1

                print (face[4])

                frame_cnn = frame[face_y1:face_y2, face_x1:face_x2]  # Get the frame for the captured head

                frame_r = cv2.resize(frame_cnn, (128, 128),
                                     interpolation=cv2.INTER_AREA)  # Resize the frame to feed into the CNN

                # roll_degree = my_head_pose_estimator.return_roll(image, radians=False)  # Evaluate the roll angle using a CNN
                # pitch_degree = my_head_pose_estimator.return_pitch(image, radians=False)  # Evaluate the pitch angle using a CNN
                # print("Estimated [roll, pitch, yaw] (degrees) ..... [" + str(roll_degree[0, 0, 0]) + "," + str(
                #     pitch_degree[0, 0, 0]) + "," + str(yaw_degree[0, 0, 0]) + "]")

                yaw_degree = my_head_pose_estimator.return_yaw(frame_r, radians=False)  # Evaluate the yaw angle using a CNN


                print ('Estimated yaw degrees..[' + str(yaw_degree[0, 0, 0]) + ']')

                # roll = my_head_pose_estimator.return_roll(image, radians=True)  # Evaluate the roll angle using a CNN
                # pitch = my_head_pose_estimator.return_pitch(image, radians=True)  # Evaluate the pitch angle using a CNN

                yaw = my_head_pose_estimator.return_yaw(frame_r, radians=True)  # Evaluate the yaw angle using a CNN

                # print("Estimated [roll, pitch, yaw] (radians) ..... [" + str(roll[0, 0, 0]) + "," + str(pitch[0, 0, 0]) + "," + str(
                #     yaw[0, 0, 0]) + "]")

                # Getting rotation and translation vector
                rot_matrix = yaw2rotmat(
                    -yaw[0, 0, 0])  # Deepgaze use different convention for the Yaw, we have to use the minus sign

                # Attention: OpenCV uses a right-handed coordinates system:
                # Looking along optical axis of the camera, X goes right, Y goes downward and Z goes forward.

                rvec, jacobian = cv2.Rodrigues(rot_matrix)
                tvec = np.array([0.0, 0.0, 1.0], np.float)  # translation vector

                zrc = yaw_degree[0, 0, 0]
                yaw_sum_cnn, yaw_avg_cnn = Exponential_smoothing_filter(face_counter, exp_sm_thres, sf, zrc, yaw_sum_cnn,
                                                                        yaw_avg_cnn)

                cognav = yaw_avg_cnn

                CPublisher.publish(abs(cognav))

                previous = now

                now = rospy.get_time()

                rospy.loginfo(now - previous)


            # Showing the frame and waiting for the exit command
            cv2.imshow('Original', frame)  # show on window

            if cv2.waitKey(1) & 0xFF == ord('q'): break  # Exit when Q is pressed

    # Release the camera
    video_capture.release()
    print("Bye...")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
