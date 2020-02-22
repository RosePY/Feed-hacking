import cv2
import numpy as np
from datetime import datetime


def get_background(frames):
	frame_stack = np.stack(frames, axis=3)
	median_frame = np.median(frame_stack, axis=3)
	return median_frame.astype(np.uint8)


def remove_object(bg_img, img, next_roi, bounding_box=True):
	frame_delta = cv2.absdiff(bg_img, img)
	img_thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
	img_dil = cv2.dilate(img_thresh, None, iterations=2)
	img_gray = cv2.cvtColor(img_dil, cv2.COLOR_BGR2GRAY)

	(x, y, w, h) = [int(v) for v in next_roi]

	if bounding_box:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	row, col = img.shape[:2]

	for i in range(y, y + h):
		for j in range(x, x + w):
			i = i if i < row else row - 1
			j = j if j < col else col - 1
			if img_gray[i, j] != 0:
				img[i, j] = bg_img[i, j]

	return img


def remove_all_objects(bg_img, img):

	frame_delta = cv2.absdiff(bg_img, img)
	img_thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
	img_dil = cv2.dilate(img_thresh, None, iterations=2)
	img = np.where(img_dil != 0, bg_img, img)

	return img

