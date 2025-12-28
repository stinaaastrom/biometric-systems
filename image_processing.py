import cv2
import os
import albumentations as A


PADDING = 20
# Use absolute paths to the model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_pretrained")
face_proto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
face_model = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")


class ImageProcesser:
        
    def __init__(self):
        self.face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

    def normalize_image(self, img):
        """Normalize image to [0, 1] float32."""
        return img.astype('float32') / 255.0

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
        face = None
        for (x1, y1, x2, y2) in face_boxes:
            face = frame[
                max(0, y1-PADDING):min(y2+PADDING, frame.shape[0]-1),
                max(0, x1-PADDING):min(x2+PADDING, frame.shape[1]-1)
            ]

        return face
    
    
    def image_enhancements(self, img, target_size=None):
        # Resize if needed
        if target_size is not None and img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)
        # Convert BGR (cv2 default) to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to YUV color space
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            # Apply CLAHE (better than simple histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
            # Convert back to RGB
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        # Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def get_augmentation_transform(self):
        """Returns comprehensive augmentation pipeline for age detection training."""
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.5
            ),
            A.CLAHE(p=0.2),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
                A.Sharpen()
            ], p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.2
            )
        ])
        return train_transform

