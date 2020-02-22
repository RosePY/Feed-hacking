import cv2
import os
import numpy as np


def get_roi(msg, img, from_center, show_crosshair):
	my_roi = cv2.selectROI(msg, img, fromCenter=from_center, showCrosshair=show_crosshair)
	cv2.destroyAllWindows()
	return my_roi


def read_roi_points(file_path, roi_path):
	filename = os.path.basename(file_path)
	file = open(roi_path, 'r')
	for line in file:
		if filename == line.split(' ')[0]:
			my_roi = line.split(' ')
			return int(my_roi[1]), int(my_roi[2]), int(my_roi[3]), int(my_roi[4].strip())
	return None
