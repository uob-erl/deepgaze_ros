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

