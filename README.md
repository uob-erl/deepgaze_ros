# deepgaze_ros
ROS wrapper package for deepgaze 

# Install deepgaze library 

https://github.com/mpatacchiola/deepgaze
Follow instructions as shown in github to install the prerequisites 

# Install dlib for python
Download dlib tar from http://dlib.net/ and 
follow instructions from http://dlib.net/compile.html for using dlib in python

# Troubleshooting

In the case of an error with cv2 check, opencv library imported in python, on terminal 

by running

```shell
python
import cv2
cv2.__version__
```

there is the chance that ros kinetic will install cv 3.3 over the 2.4 required by the algorithm to perform properly 

# Implementation on Ros

We are going with the solution of running the project on two machines 
in order to run the script for the publisher you will need the deepgaze repo
and point to the correct locations of the repo on the script 

Place the shape_predictor_68_face_landmarks.dat in the etc folder of deepgaze 
and declare the path to the etc folder of deepgaze on your system. 

```shell
 # Declaring the two classifiers
    my_cascade = haarCascade("../etc/xml/haarcascade_frontalface_alt.xml", "../etc/xml/haarcascade_profileface.xml")
    # TODO If missing, example file can be retrieved from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    my_detector = faceLandmarkDetection('../etc/shape_predictor_68_face_landmarks.dat')
```
