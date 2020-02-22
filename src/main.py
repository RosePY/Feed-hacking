import cv2
import roi
import background as bg
import argparse


def main():
	opencv_object_trackers = {
		'csrt': cv2.TrackerCSRT_create(),
		'kcf': cv2.TrackerKCF_create(),
		'boosting': cv2.TrackerBoosting_create(),
		'mil': cv2.TrackerMIL_create(),
		'tld': cv2.TrackerTLD_create(),
		'medianflow': cv2.TrackerMedianFlow_create(),
		'mosse': cv2.TrackerMOSSE_create()
	}

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str, help="Path to input video file")
	ap.add_argument("-o", "--output", type=str, help="Path to output video file")
	ap.add_argument("-d", "--detect", type=bool, default=False, help="Detect target in video (0 | 1)")
	ap.add_argument("-t", "--tracker", type=str, default=None, help="Choose the tracker algorithm")
	ap.add_argument("-r", "--region", type=str, default=None, help="Path to the ROI points (.txt)")
	ap.add_argument("-b", "--bounding_box", type=bool, default=False, help="Draw bounding box in output video")
	ap.add_argument("-p", "--play", type=bool, default=False, help="Play video")
	ap.add_argument("-g", "--background", type=str, default=None, help="Read background")

	args = vars(ap.parse_args())

	print('Processing file:', args['input'])

	if not (args.get("input", False) and args.get("output", False)):
		print("Usage: python main.py -i path_to_video [-t detect target]")
		exit(0)

	# initializing capture of video
	cap = cv2.VideoCapture(args['input'])

	# number of frames of video
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# number of frames per second
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	frames = []

	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		frames.append(frame)
		#print('Reading frame', i + 1, 'of', n_frames, end='\r')
		i += 1
	cap.release()

	n_frames = i

	print("Extracting background ...")

	if args['background'] is None:
		bg_image = bg.get_background(frames[:n_frames])
	else:
		bg_image = cv2.imread(args['background'])

	out = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'XVID'), fps, (bg_image.shape[1], bg_image.shape[0]))

	if not args['detect']:
		for ix in range(n_frames):
			#print('Processing frame', ix + 1, 'of', n_frames, end='\r')
			frame_wo_objects = bg.remove_all_objects(bg_image, frames[ix])
			out.write(frame_wo_objects)

			if cv2.waitKey(int((1 / fps) * 1000)) == ord('q'):
				cv2.destroyAllWindows()
				break
	elif args['region'] is None:
		ix = 0
		while True:
			cv2.namedWindow('Press key p to pause the video', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('Press key p to pause the video', 960, 720)
			cv2.imshow('Press key p to pause the video', frames[ix])
			key = cv2.waitKey(int((1 / fps) * 1000))

			if True or key == ord('p'):
				curr_ix = ix
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				exit(0)

			ix = (ix + 1) % n_frames

		curr_frame = frames[curr_ix]

		cv2.namedWindow('Select object to track', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Select object to track', 960, 720)
		my_roi = roi.get_roi('Select object to track', frames[curr_ix], False, False)

		tracker = opencv_object_trackers[args['tracker']]
		tracker.init(curr_frame, my_roi)

		for ix in range(curr_ix + 1, n_frames):
			#print('Processing frame', ix + 1, 'of', n_frames, end='\r')
			success, next_roi = tracker.update(frames[ix])

			if success:
				object_wo = bg.remove_object(bg_image, frames[ix], next_roi, bounding_box=args['bounding_box'])
				out.write(object_wo)
				if args['play']:
					cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
					cv2.resizeWindow('Output', 960, 720)
					cv2.imshow('Output', frames[ix])
					if cv2.waitKey(int((1 / fps) * 1000)) == ord('q'):
						cv2.destroyAllWindows()
						break
	else:
		my_roi = roi.read_roi_points(args['input'], args['region'])
		tracker = opencv_object_trackers[args['tracker']]
		tracker.init(frames[0], my_roi)

		for ix in range(1, n_frames):
			#print('Processing frame', ix + 1, 'of', n_frames, end='\r')
			success, next_roi = tracker.update(frames[ix])

			if success:
				object_wo = bg.remove_object(bg_image, frames[ix], next_roi, bounding_box=args['bounding_box'])
				out.write(object_wo)
				if args['play']:
					cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
					cv2.resizeWindow('Output', 960, 720)
					cv2.imshow('Output', frames[ix])
					if cv2.waitKey(int((1 / fps) * 1000)) == ord('q'):
						cv2.destroyAllWindows()
						break

	print('Output file generated:', args['output'])
	out.release()


if __name__ == "__main__":
	main()
