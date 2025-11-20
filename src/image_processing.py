import cv2
import os

PADDING = 20
# Use absolute paths to the model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_pretrained")
face_proto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
face_model = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")


class ImageProcesser:
    def __init__(self):
        self.face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

    def detect_crop_faces(self, frame, conf_threshold=0.7):
        """Face detetion using deep learning, image based technique."""
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)

        # Crop image with padding        
        for (x1, y1, x2, y2) in face_boxes:
            face = frame[
                max(0, y1-PADDING):min(y2+PADDING, frame.shape[0]-1),
                max(0, x1-PADDING):min(x2+PADDING, frame.shape[1]-1)
            ]

        return face
    
    
    def image_enhancements(self, img):
        # Convert to YUV color space
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        # Apply CLAHE (better than simple histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        
        # Convert back to BGR
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
