import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, math, random, time
import numpy as np

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

COLOR_WHITE=(255,255,255); COLOR_BLACK=(0,0,0); COLOR_GREEN=(0,220,0)
COLOR_RED=(0,0,220); COLOR_YELLOW=(0,255,255); COLOR_CYAN=(255,255,0)
COLOR_MAGENTA=(255,0,255); COLOR_ORANGE=(0,165,255); COLOR_BG=(40,40,40)

def draw_hand_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks]
        for s,e in HAND_CONNECTIONS:
            cv2.line(frame, pts[s], pts[e], (255,0,0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0,255,0), -1)

def is_finger_stretched(hlm, tip, pip):
    return hlm[tip].y < hlm[pip].y

def detect_gesture(hlm):
    fingers = [is_finger_stretched(hlm,t,p) for t,p in [(8,6),(12,10),(16,14),(20,18)]]
    if all(fingers): return "PAPER"
    if not any(fingers): return "FIST"
    return "OTHER"

def get_hand_center(hlm):
    return sum(l.x for l in hlm)/len(hlm), sum(l.y for l in hlm)/len(hlm)

class Mosquito:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x = random.randint(x_min + 40, x_max - 40)
        self.y = random.randint(y_min + 40, y_max - 40)
        self.vx = random.choice([-1,1]) * random.uniform(3,6)
        self.vy = random.choice([-1,1]) * random.uniform(3,6)
        self.wing_phase = random.uniform(0, 6.28)
        self.timer = 0; self.size = 18; self.alive = True

    def update(self):
        if not self.alive: return
        self.timer += 1
        if self.timer % 20 == 0:
            self.vx += random.uniform(-3,3); self.vy += random.uniform(-3,3)
        spd = math.sqrt(self.vx**2 + self.vy**2)
        if spd > 7: self.vx, self.vy = self.vx/spd*7, self.vy/spd*7
        if spd < 2: self.vx *= 1.5; self.vy *= 1.5
        self.x += self.vx; self.y += self.vy
        m = 40
        if self.x < self.x_min+m: self.x=self.x_min+m; self.vx=abs(self.vx)
        if self.x > self.x_max-m: self.x=self.x_max-m; self.vx=-abs(self.vx)
        if self.y < self.y_min+m: self.y=self.y_min+m; self.vy=abs(self.vy)
        if self.y > self.y_max-m: self.y=self.y_max-m; self.vy=-abs(self.vy)
        self.wing_phase += 0.5

    def draw(self, frame):
        if not self.alive: return
        cx, cy, s = int(self.x), int(self.y), self.size
        cv2.ellipse(frame,(cx,cy),(s//3,s),0,0,360,(30,30,50),-1)
        cv2.ellipse(frame,(cx,cy),(s//3,s),0,0,360,(80,80,120),1)
        cv2.circle(frame,(cx,cy-s-4),s//3,(20,20,40),-1)
        cv2.circle(frame,(cx-3,cy-s-5),2,(0,0,200),-1)
        cv2.circle(frame,(cx+3,cy-s-5),2,(0,0,200),-1)
        cv2.line(frame,(cx,cy-s-7),(cx,cy-s-16),(60,60,100),1)
        wo = int(math.sin(self.wing_phase)*12)
        for sign, woff in [(-1, wo), (1, -wo)]:
            pts = np.array([[cx+sign*2,cy-4],[cx+sign*(s+10),cy-8+woff],
                            [cx+sign*(s+5),cy+4+woff]], np.int32)
            cv2.fillPoly(frame,[pts],(180,180,200))
            cv2.polylines(frame,[pts],True,(120,120,150),1)
        for i in range(3):
            for sign in [-1, 1]:
                cv2.line(frame,(cx+sign*3,cy+2+i*4),
                         (cx+sign*(8+i*2),cy+self.size//2+i*3),(50,50,80),1)

def draw_hud(frame, p1_score, p2_score, win_target, boundary_x):
    h, w, _ = frame.shape
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,65),COLOR_BG,-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    cv2.putText(frame,"MOSQUITO CATCH!",(w//2-160,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,COLOR_CYAN,2)
    cv2.putText(frame,f"First to {win_target}!",(w//2-55,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_YELLOW,1)
    ov2 = frame.copy()
    cv2.rectangle(ov2,(0,65),(w,105),(30,30,30),-1)
    cv2.addWeighted(ov2,0.7,frame,0.3,0,frame)
    cv2.putText(frame,f"P1: {p1_score}",(20,95),cv2.FONT_HERSHEY_SIMPLEX,0.8,COLOR_GREEN,2)
    cv2.putText(frame,f"P2: {p2_score}",(w-140,95),cv2.FONT_HERSHEY_SIMPLEX,0.8,COLOR_MAGENTA,2)
    for y_pos in range(110, h-50, 12):
        cv2.line(frame,(boundary_x,y_pos),(boundary_x,min(y_pos+6,h-50)),(0,0,255),2)
    cv2.putText(frame,"BOUNDARY",(boundary_x-45,h-55),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    cv2.putText(frame,"<-- P1",(20,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR_GREEN,1)
    cv2.putText(frame,"P2 -->",(w-110,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR_MAGENTA,1)
    ov3 = frame.copy()
    cv2.rectangle(ov3,(0,h-40),(w,h),COLOR_BG,-1)
    cv2.addWeighted(ov3,0.7,frame,0.3,0,frame)
    cv2.putText(frame,"R=Reset Q=Quit | Fist=GRAB! Stay on YOUR side!",
                (10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.43,(150,150,150),1)

def draw_catch_effect(frame, cx, cy, progress):
    r = int(20 + progress*60)
    a = max(0, 1.0 - progress)
    cv2.circle(frame,(cx,cy),r,(0,int(255*a),int(255*a)),2)
    if progress < 0.5:
        cv2.putText(frame,"SPLAT!",(cx-40,cy-30),cv2.FONT_HERSHEY_SIMPLEX,1.0,COLOR_RED,3)

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: No camera"); return
    ret, tf = cap.read()
    if not ret: print("Error: No frame"); return
    FH, FW = tf.shape[:2]
    MID_X = FW // 2  # boundary line
    CATCH_RADIUS = 90; WIN_TARGET = 10
    BOUND_PAD = 100  # keep mosquitos this far from the center boundary

    print("=== MOSQUITO CATCH GAME ===")
    print("P1=Left side, P2=Right side. Each has their own mosquito.")
    print("Stay on YOUR side! Cross the boundary = opponent gets a point!")
    print(f"First to {WIN_TARGET} wins! R=Reset | Q=Quit")

    # State
    p1_score=0; p2_score=0
    game_state="countdown"; countdown_start=time.time(); countdown_val=3
    mosq_p1 = Mosquito(0, MID_X - BOUND_PAD, 110, FH-50)
    mosq_p2 = Mosquito(MID_X + BOUND_PAD, FW, 110, FH-50)
    catch_winner=None; catch_time=0; catch_pos=(0,0)
    foul_player=None; foul_time=0
    frame_ts=0
    # Track splat effects (list of (x, y, time))
    splat_effects = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)
        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks)

        # Identify players
        p1_gesture=None; p2_gesture=None; p1_center=None; p2_center=None
        if result.hand_landmarks:
            hands = []
            for hlm in result.hand_landmarks:
                cx, cy = get_hand_center(hlm)
                hands.append((cx, cy, detect_gesture(hlm)))
            hands.sort(key=lambda h: h[0])
            if len(hands) >= 1:
                p1_gesture = hands[0][2]
                p1_center = (int(hands[0][0]*FW), int(hands[0][1]*FH))
            if len(hands) >= 2:
                p2_gesture = hands[-1][2]
                p2_center = (int(hands[-1][0]*FW), int(hands[-1][1]*FH))

        # Draw any active splat effects
        for sx, sy, st in splat_effects:
            prog = min((time.time()-st)/0.8, 1.0)
            draw_catch_effect(frame, sx, sy, prog)
        splat_effects = [(sx,sy,st) for sx,sy,st in splat_effects if time.time()-st < 0.8]

        # ── STATE MACHINE ──
        if game_state == "countdown":
            elapsed = time.time() - countdown_start
            countdown_val = 3 - int(elapsed)
            mosq_p1.draw(frame); mosq_p2.draw(frame)
            draw_hud(frame,p1_score,p2_score,WIN_TARGET,MID_X)

            # Check both players are showing PAPER during countdown
            p1_not_paper = p1_gesture is not None and p1_gesture != "PAPER"
            p2_not_paper = p2_gesture is not None and p2_gesture != "PAPER"

            if p1_not_paper and p2_not_paper:
                foul_player = "BOTH"
                foul_time = time.time()
                game_state = "foul"
            elif p1_not_paper:
                foul_player = "P1"
                p2_score += 1
                foul_time = time.time()
                game_state = "foul"
            elif p2_not_paper:
                foul_player = "P2"
                p1_score += 1
                foul_time = time.time()
                game_state = "foul"
            elif countdown_val > 0:
                cv2.putText(frame,f"Hands OPEN... {countdown_val}",(FW//2-160,FH//2),
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,COLOR_YELLOW,3)
                # Show green checkmarks for players holding paper
                if p1_gesture == "PAPER":
                    cv2.putText(frame,"P1: READY",(20,FH//2+40),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,COLOR_GREEN,2)
                if p2_gesture == "PAPER":
                    cv2.putText(frame,"P2: READY",(FW-200,FH//2+40),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,COLOR_MAGENTA,2)
            else:
                game_state = "playing"

        elif game_state == "playing":
            mosq_p1.update(); mosq_p2.update()
            mosq_p1.draw(frame); mosq_p2.draw(frame)
            draw_hud(frame,p1_score,p2_score,WIN_TARGET,MID_X)

            # Draw catch zones around mosquitos
            if mosq_p1.alive:
                cv2.circle(frame,(int(mosq_p1.x),int(mosq_p1.y)),CATCH_RADIUS,(50,80,50),1)
            if mosq_p2.alive:
                cv2.circle(frame,(int(mosq_p2.x),int(mosq_p2.y)),CATCH_RADIUS,(80,50,80),1)

            # Show gestures
            if p1_gesture:
                icon = "GRAB!" if p1_gesture=="FIST" else ("OPEN" if p1_gesture=="PAPER" else "???")
                cv2.putText(frame,f"P1: {icon}",(20,FH-60),cv2.FONT_HERSHEY_SIMPLEX,0.7,COLOR_GREEN,2)
            if p2_gesture:
                icon = "GRAB!" if p2_gesture=="FIST" else ("OPEN" if p2_gesture=="PAPER" else "???")
                cv2.putText(frame,f"P2: {icon}",(FW-200,FH-60),cv2.FONT_HERSHEY_SIMPLEX,0.7,COLOR_MAGENTA,2)

            # ── CHECK BOUNDARY FOULS ──
            p1_foul = p1_center and p1_center[0] > MID_X
            p2_foul = p2_center and p2_center[0] < MID_X

            if p1_foul and p2_foul:
                # Both crossed — both lose, no score
                foul_player = "BOTH"
                foul_time = time.time()
                game_state = "foul"
            elif p1_foul:
                foul_player = "P1"
                p2_score += 1
                foul_time = time.time()
                game_state = "foul"
            elif p2_foul:
                foul_player = "P2"
                p1_score += 1
                foul_time = time.time()
                game_state = "foul"
            else:
                # ── CHECK CATCHES ──
                if p1_gesture=="FIST" and p1_center and mosq_p1.alive:
                    d = math.sqrt((p1_center[0]-mosq_p1.x)**2+(p1_center[1]-mosq_p1.y)**2)
                    cv2.circle(frame,p1_center,CATCH_RADIUS,
                               COLOR_GREEN if d<CATCH_RADIUS else (50,100,50),2)
                    if d < CATCH_RADIUS:
                        p1_score += 1
                        splat_effects.append((int(mosq_p1.x),int(mosq_p1.y),time.time()))
                        mosq_p1 = Mosquito(0, MID_X - BOUND_PAD, 110, FH-50)

                if p2_gesture=="FIST" and p2_center and mosq_p2.alive:
                    d = math.sqrt((p2_center[0]-mosq_p2.x)**2+(p2_center[1]-mosq_p2.y)**2)
                    cv2.circle(frame,p2_center,CATCH_RADIUS,
                               COLOR_MAGENTA if d<CATCH_RADIUS else (80,50,80),2)
                    if d < CATCH_RADIUS:
                        p2_score += 1
                        splat_effects.append((int(mosq_p2.x),int(mosq_p2.y),time.time()))
                        mosq_p2 = Mosquito(MID_X + BOUND_PAD, FW, 110, FH-50)

                # Check win
                if p1_score >= WIN_TARGET or p2_score >= WIN_TARGET:
                    game_state = "gameover"

        elif game_state == "foul":
            draw_hud(frame,p1_score,p2_score,WIN_TARGET,MID_X)
            elapsed = time.time() - foul_time
            if int(elapsed*6) % 2 == 0:
                ov = frame.copy()
                cv2.rectangle(ov,(0,0),(FW,FH),(0,0,180),-1)
                cv2.addWeighted(ov,0.3,frame,0.7,0,frame)
            if foul_player == "BOTH":
                txt = "BOTH CROSSED! No point!"; c = COLOR_YELLOW
            elif foul_player == "P1":
                txt = "P1 FOUL! P2 +1"; c = COLOR_MAGENTA
            else:
                txt = "P2 FOUL! P1 +1"; c = COLOR_GREEN
            cv2.putText(frame,"FOUL!",(FW//2-70,FH//2-30),cv2.FONT_HERSHEY_SIMPLEX,1.5,COLOR_RED,4)
            cv2.putText(frame,txt,(FW//2-160,FH//2+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,c,2)
            if elapsed > 1.5:
                if p1_score >= WIN_TARGET or p2_score >= WIN_TARGET:
                    game_state = "gameover"
                else:
                    mosq_p1 = Mosquito(0, MID_X - BOUND_PAD, 110, FH-50)
                    mosq_p2 = Mosquito(MID_X + BOUND_PAD, FW, 110, FH-50)
                    game_state = "countdown"; countdown_start = time.time()

        elif game_state == "gameover":
            draw_hud(frame,p1_score,p2_score,WIN_TARGET,MID_X)
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(FW,FH),COLOR_BLACK,-1)
            cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
            if p1_score >= WIN_TARGET:
                winner, wc = "PLAYER 1 WINS!", COLOR_GREEN
            else:
                winner, wc = "PLAYER 2 WINS!", COLOR_MAGENTA
            cv2.putText(frame,"GAME OVER",(FW//2-130,FH//2-40),cv2.FONT_HERSHEY_SIMPLEX,1.2,COLOR_WHITE,2)
            cv2.putText(frame,winner,(FW//2-180,FH//2+20),cv2.FONT_HERSHEY_SIMPLEX,1.2,wc,3)
            cv2.putText(frame,f"Score: {p1_score} - {p2_score}",(FW//2-100,FH//2+70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,COLOR_WHITE,2)
            cv2.putText(frame,"R = Play again | Q = Quit",(FW//2-160,FH//2+120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(150,150,150),1)

        cv2.imshow("Mosquito Catch!", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            p1_score=0; p2_score=0
            game_state="countdown"; countdown_start=time.time()
            mosq_p1 = Mosquito(0, MID_X - BOUND_PAD, 110, FH-50)
            mosq_p2 = Mosquito(MID_X + BOUND_PAD, FW, 110, FH-50)

    landmarker.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
