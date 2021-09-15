import numpy as np
import cv2
from PIL import Image
import warnings
import winsound
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
import time
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Configure path to trained SSD Model and mable map file
#PATH_TO_SAVED_MODEL = "D:\\Driver_drowsiness_detection\\my_ssd_mobilenet_v2_fpnlite\\saved_model"
#PATH_TO_LABEL_MAP = "C:\\Users\\gijop\\Downloads\\MS\\TensorFlow\\workspace\\training_demo\\annotations\\label_map.pbtxt"
PATH_TO_LABEL_MAP = "label_map.pbtxt"
PATH_TO_SAVED_MODEL = "saved_model"
print('Loading model...', end='')
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
# Loading the label_map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_MAP, use_display_name=True)

VIDEO_FILE_PATH = 'WebcamRecord'
timestr = time.strftime("%Y%m%d-%H%M%S")
SAVE_VIDEO_FILE_PATH = VIDEO_FILE_PATH + "_" + timestr + ".avi"

# capture object acquire video from webcam
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(SAVE_VIDEO_FILE_PATH, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))

# setting for drowsiness alert text
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (25, 50)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

#parameters for calibrating drowsiness detection
MIN_SCORE_THRESHOLD = 0.6
DROWSINESS_DETECT_THRESHOLD = timedelta(seconds=5) # to trigger drowsiness alert after 5 seconds
DETECTION_CLASS_OPEN = 1
DETECTION_CLASS_CLOSED = 2

Eyes_flgClosed = 0 		# Status 'Eyes Closed event'
Eyes_tiClosed = datetime.now()  ## Time stamp - the start of 'Eyes Closed event'

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)  # running inference using trained SSD model
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = frame.copy()
    
    # it is observed that at low light conditions false 'closed' conditions are detected (large bounding box)
    # a filter logic based on the area of the detected bounding box is used for avoiding false drowsiness detection
    h1, w1, h2, w2 = detections['detection_boxes'][0]
    Area_DetnBox =  (h2-h1)*(w2-w1)
    if(Area_DetnBox < 0.5):    
        # drawing bounding box for detected objects
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh= MIN_SCORE_THRESHOLD,
            agnostic_mode=False)        

        # consider eye 'closed, detection_classes =2' detections with minimum score threshold
		# set the flgClosed and record time (only time stamp of first closed event is recorded)
        if ((detections['detection_scores'][0] > MIN_SCORE_THRESHOLD) 
                and (detections['detection_classes'][0] == DETECTION_CLASS_CLOSED)
                and (Eyes_flgClosed == 0)):
            Eyes_flgClosed = 1 
            Eyes_tiClosed = datetime.now() 
		# reset the 
        elif ((detections['detection_scores'][0] > MIN_SCORE_THRESHOLD)
                and (detections['detection_classes'][0] == DETECTION_CLASS_OPEN)):
            Eyes_flgClosed = 0        
       
	   # if the ''Eyes Closed event' prolongs for more than 5 seconds, generate audible drowsiness alert
        if ((Eyes_flgClosed == 1) and ((datetime.now() - Eyes_tiClosed) > DROWSINESS_DETECT_THRESHOLD)):
            cv2.putText(image_np_with_detections, 'Drowsiness detected!!!', bottomLeftCornerOfText, font, fontScale,
                        fontColor, lineType) # display Drowsiness alert
            winsound.PlaySound("sound2.wav", winsound.SND_ASYNC) # play drowsiness alarm

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # quit from session

    out.write(image_np_with_detections) # record the frame with bounding boxes and drosiness alert
    cv2.imshow('frame', image_np_with_detections) # display frame

# When everything done, release the capture and recorder
cap.release()
out.release()
cv2.destroyAllWindows()




