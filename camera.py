import time
import cv2
import numpy as np
import math
import random

# Global list to store particles
particles = []

class Particle:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-3, 3)  # Random horizontal velocity
        self.vy = random.uniform(-4, -1)  # Random upward velocity
        self.gravity = 0.15  # Gravity acceleration
        self.creation_time = time.time()
        self.lifetime = 1.0  # Disappear after 1 second
        self.color = (random.randint(50, 150), random.randint(150, 255), random.randint(150, 255))
        self.size = random.randint(2, 5)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity  # Apply gravity
    
    def is_alive(self):
        return time.time() - self.creation_time < self.lifetime
    
    def draw(self, frame):
        age = time.time() - self.creation_time
        alpha = 1.0 - (age / self.lifetime)  # Fade out
        color = tuple(int(c * alpha) for c in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, color, -1)

def main():
    # Initialize the webcam
    global particles
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully. Press 'q' to quit.")
    cx,cy = 0,0
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        b = frame[:,:,0]
        g = frame[:,:,1]

        r = frame[:,:,2] 

        ret, b = cv2.threshold(b, 30, 255, cv2.THRESH_BINARY_INV)
        ret, g = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
        ret ,r = cv2.threshold(r, 35, 255, cv2.THRESH_BINARY_INV)

        # Concatenate frames: frame, g on top row; r, b on bottomqq row
        top_row = np.hstack([frame, g[:,:,np.newaxis].repeat(3, axis=2)])
        bottom_row = np.hstack([r[:,:,np.newaxis].repeat(3, axis=2), b[:,:,np.newaxis].repeat(3, axis=2)])
        concatenated = np.vstack([top_row, bottom_row])

        final_mask = cv2.bitwise_and(b,g)
        final_mask = cv2.bitwise_and(r,final_mask)
        cv2.imshow("mask", final_mask)
        contours, _h = cv2.findContours(final_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # print(len(contours))

        max_contour = None
        for cnt in contours:
            if cnt.shape[0] < 100:
                continue
            if max_contour is None or max_contour.shape[0] < cnt.shape[0]:
                max_contour = cnt
        # print(max_contour)
        moment = cv2.moments(max_contour)
        try:
            cx,cy = int(moment["m10"]/moment["m00"]),int(moment["m01"]/moment["m00"])
            print(cx,cy)
        except ZeroDivisionError:
            print("error, wand too far away or not in frame.")
            print("range: apoxx.3 - 15cm")
            pass

        

        frame = frame * 0.5 + np.ones_like(frame) * 127
        frame = frame.astype(np.uint8)
        # Draw magic wand at the location
        frame = cv2.circle(frame, (cx,cy), 10, (0,0,255), 2)
        
        # Draw magic wand stick (line)
        wand_length = 100
        wand_angle = 90  # degrees
        angle_rad = math.radians(wand_angle)
        end_x = int(cx + wand_length * math.cos(angle_rad))
        end_y = int(cy - wand_length * math.sin(angle_rad))
        cv2.line(frame, (cx, cy), (end_x, end_y), (200, 100, 255), 3)
        
        # Create particles at the wand tip
        for _ in range(random.randint(2, 5)):
            particles.append(Particle(end_x, end_y))
        
        # Update and draw particles
        particles = [p for p in particles if p.is_alive()]
        for p in particles:
            p.update()
            p.draw(frame)
        
        # Draw magic wand star at the tip
        star_size = 30
        pts = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            radius = star_size if i % 2 == 0 else star_size // 2
            x = int(end_x + radius * math.cos(angle))
            y = int(end_y + radius * math.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(frame, [pts], (100, 200, 255))
        cv2.polylines(frame, [pts], True, (50, 100, 255), 2)
        
        cv2.imshow("location", frame)
        # Resize to 3104800 * 1920
        resized = cv2.resize(concatenated, (1920,1080))
        
        cv2.imshow("Concatenated and Resized", resized)
        
        # print(frame.shape)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break


    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
