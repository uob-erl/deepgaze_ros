# deepgaze_ros
ROS wrapper package for deepgaze 

# Install deepgaze library 

https://github.com/mpatacchiola/deepgaze
Follow instructions as shown in github to install the prerequisites 

# Install dlib for python
Download dlib tar from http://dlib.net/ and 
follow instructions from http://dlib.net/compile.html for using dlib in python

In the case of an error while running check opencv library imported in python 

by running 

python
import cv2
cv2.__version__

on terminal 

there is the chance that ros kinetic will install cv 3.3 over the 2.4 required by the algorithm to perform properly 

