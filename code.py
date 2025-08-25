import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import pygame
import time
from collections import deque

class ExamProctoringSystem:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7)  # model_selection=1 for full-range detection
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Initialize parameters
        self.face_out_of_frame_count = 0
        self.phone_detected_count = 0
        self.multiple_faces_count = 0
        self.total_warnings = 0
        self.max_warnings = 5
        
        # For gaze detection
        self.eye_indices = {
            'left': [33, 160, 158, 133, 153, 144],  # Left eye landmarks
            'right': [362, 385, 387, 263, 373, 380]  # Right eye landmarks
        }
        
        # For focus history
        self.focus_history = deque(maxlen=30)  # Track focus over last 30 frames
        
        # Initialize audio alert
        pygame.mixer.init()
        # Create a simple beep sound if warning.wav doesn't exist
        try:
            self.warning_sound = pygame.mixer.Sound("warning.wav")
        except:
            # Create a simple beep sound
            sound_array = np.zeros(44100)
            for i in range(44100):
                if i % 100 < 50:
                    sound_array[i] = 0.5
            self.warning_sound = pygame.mixer.Sound(buffer=bytearray((sound_array * 127).astype(np.int8)))
        
        # Face position thresholds (relative to frame size)
        self.face_position_threshold = 0.2  # 20% from edges
        
    def eye_aspect_ratio(self, eye_landmarks, frame_shape):
        # Get the eye landmarks coordinates
        eye_points = []
        for idx in eye_landmarks:
            landmark = self.landmarks.landmark[idx]
            x = int(landmark.x * frame_shape[1])
            y = int(landmark.y * frame_shape[0])
            eye_points.append((x, y))
        
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear, eye_points
    
    def detect_phone(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phone_detected = False
        
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if the contour has 4 vertices (rectangle)
            if len(approx) == 4:
                # Compute the bounding box of the contour
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if the aspect ratio is similar to a phone
                aspect_ratio = w / float(h)
                if 0.8 <= aspect_ratio <= 1.8 and w > 100 and h > 100:
                    # Check if it's not in the face region (to avoid false positives)
                    if not self.is_in_face_region(x, y, w, h, frame.shape):
                        phone_detected = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "PHONE DETECTED", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return phone_detected, frame
    
    def is_in_face_region(self, x, y, w, h, frame_shape):
        # Check if the detected rectangle overlaps with face regions
        if not hasattr(self, 'face_bbox'):
            return False
            
        face_x, face_y, face_w, face_h = self.face_bbox
        if (x < face_x + face_w and x + w > face_x and 
            y < face_y + face_h and y + h > face_y):
            return True
        return False
    
    def check_face_position(self, frame_shape):
        # Check if face is too close to the edges of the frame
        if not hasattr(self, 'face_bbox'):
            return False
            
        x, y, w, h = self.face_bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate distances to edges
        left_dist = x
        right_dist = frame_w - (x + w)
        top_dist = y
        bottom_dist = frame_h - (y + h)
        
        # Check if face is too close to any edge
        threshold = min(frame_w, frame_h) * self.face_position_threshold
        if min(left_dist, right_dist, top_dist, bottom_dist) < threshold:
            return True
        return False
    
    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Detection
        face_detection_results = self.face_detection.process(rgb_frame)
        
        # Reset face detection flags
        face_detected = False
        multiple_faces = False
        
        # Check for faces
        if face_detection_results.detections:
            face_detected = True
            multiple_faces = len(face_detection_results.detections) > 1
            
            # Get the first (primary) face
            detection = face_detection_results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            face_w = int(bboxC.width * w)
            face_h = int(bboxC.height * h)
            
            self.face_bbox = (x, y, face_w, face_h)
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
            
            # Check face position
            if self.check_face_position(frame.shape):
                cv2.putText(frame, "MOVE TO CENTER", (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Check for multiple faces
        if multiple_faces:
            self.multiple_faces_count += 1
            cv2.putText(frame, "MULTIPLE FACES DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.multiple_faces_count = max(0, self.multiple_faces_count - 1)
        
        # Check if face is detected
        if not face_detected:
            self.face_out_of_frame_count += 1
            cv2.putText(frame, "FACE NOT DETECTED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.focus_history.append(False)
        else:
            self.face_out_of_frame_count = max(0, self.face_out_of_frame_count - 0.5)
            
            # Process the frame with MediaPipe Face Mesh for more detailed analysis
            face_mesh_results = self.face_mesh.process(rgb_frame)
            
            if face_mesh_results.multi_face_landmarks:
                self.landmarks = face_mesh_results.multi_face_landmarks[0]
                
                # Check gaze direction
                left_ear, left_eye_points = self.eye_aspect_ratio(self.eye_indices['left'], frame.shape)
                right_ear, right_eye_points = self.eye_aspect_ratio(self.eye_indices['right'], frame.shape)
                
                # Average the eye aspect ratio
                ear = (left_ear + right_ear) / 2.0
                
                # Check if eyes are open
                eyes_open = ear > 0.2
                self.focus_history.append(eyes_open)
                
                # Draw eye landmarks
                for point in left_eye_points + right_eye_points:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Detect phones
        phone_detected, frame = self.detect_phone(frame)
        if phone_detected:
            self.phone_detected_count += 1
        else:
            self.phone_detected_count = max(0, self.phone_detected_count - 0.5)
        
        # Calculate focus score (percentage of recent frames with eyes open)
        if len(self.focus_history) > 0:
            focus_score = sum(self.focus_history) / len(self.focus_history)
            if focus_score < 0.7:
                cv2.putText(frame, f"LOW FOCUS: {focus_score:.2%}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Check if warnings should be issued
        warnings = []
        if self.face_out_of_frame_count > 15:  
            warnings.append("Keep your face in the frame")
            self.face_out_of_frame_count = 0
            self.total_warnings += 1
        
        if self.phone_detected_count > 30:  # About 1 second at 30 FPS
            warnings.append("Phone usage detected")
            self.phone_detected_count = 0
            self.total_warnings += 1
        
        if self.multiple_faces_count > 30:  # About 1 second at 30 FPS
            warnings.append("Multiple people detected")
            self.multiple_faces_count = 0
            self.total_warnings += 1
        
        # Display warnings
        if warnings:
            self.warning_sound.play()
            for i, warning in enumerate(warnings):
                cv2.putText(frame, f"WARNING: {warning}", (10, 120 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display warning count
        cv2.putText(frame, f"Warnings: {self.total_warnings}/{self.max_warnings}", 
                   (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if self.total_warnings > 0 else (0, 255, 0), 2)
        
        # Check if exam should be terminated
        if self.total_warnings >= self.max_warnings:
            cv2.putText(frame, "EXAM TERMINATED - Too many warnings", 
                       (frame.shape[1] // 2 - 200, frame.shape[0] // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Exam Proctoring", frame)
            cv2.waitKey(3000)  # Show message for 3 seconds
            return True, frame
        
        return False, frame

def main():
    # Initialize the proctoring system
    proctor = ExamProctoringSystem()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Exam proctoring system started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        exam_terminated, processed_frame = proctor.process_frame(frame)
        
        # Display the processed frame
        cv2.imshow("Exam Proctoring", processed_frame)
        
        # Check if exam is terminated or user wants to quit
        if exam_terminated:
            print("Exam terminated due to multiple violations.")
            break
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()