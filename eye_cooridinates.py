import cv2

def get_eyes_coordinates(image,detection_result):
    for detection in detection_result.detections:
        right_eye_x=int(detection_result.detections[0].keypoints[0].x*image.shape[1])-3
        left_eye_x=int(detection_result.detections[0].keypoints[1].x*image.shape[1])+3
        right_eye_y=int(detection_result.detections[0].keypoints[0].y*image.shape[0])-3
        left_eye_y=int(detection_result.detections[0].keypoints[1].y*image.shape[0])-7
        x=int((left_eye_x-right_eye_x)/2 )
        y=x
        image1=cv2.rectangle(image,(max(0,right_eye_x-x),max(0,right_eye_y-y)),(min(image.shape[1]-1,max(0,right_eye_x-x)+2*x),min(max(0,right_eye_y-y)+2*y,image.shape[0]-1)),(255,0,0),2)
        image1=cv2.rectangle(image,(max(0,left_eye_x-x),max(0,left_eye_y-y)),(min(image.shape[1]-1,max(0,left_eye_x-x)+2*x),min(max(0,left_eye_y-y)+2*y,image.shape[0]-1)),(255,0,0),2)
        right_eye=image[max(0,right_eye_y-y):min(max(0,right_eye_y-y)+2*y,image.shape[0]-1),max(0,right_eye_x-x):min(image.shape[1]-1,max(0,right_eye_x-x)+2*x)]
        left_eye=image[max(0,left_eye_y-y):min(max(0,left_eye_y-y)+2*y,image.shape[0]-1),max(0,left_eye_x-x):min(image.shape[1]-1,max(0,left_eye_x-x)+2*x)]

        left_eye = cv2.resize((cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)),(60,60))
        right_eye = cv2.resize((cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)),(60,60))
        return left_eye,right_eye,image1