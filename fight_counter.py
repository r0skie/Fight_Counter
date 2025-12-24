import cv2
import numpy as np
import mss
import time
from pynput import mouse

# ================= CONFIG =================
TEMPLATE_PATH = "fight.png"
THRESHOLD = 0.85          # tingkat kecocokan
SCAN_DELAY = 0.3          # detik
MONITOR_INDEX = 1         # layar utama
# =========================================

# Load template
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError("fight.png tidak ditemukan")

w, h = template.shape[::-1]

sct = mss.mss()
monitor = sct.monitors[MONITOR_INDEX]

fight_box = None
fight_click_count = 0

print("[INFO] Fight Click Counter started")
print("[INFO] Tekan CTRL+C untuk keluar\n")

def on_click(x, y, button, pressed):
    global fight_click_count, fight_box

    if pressed and fight_box:
        fx, fy, fw, fh = fight_box
        if fx <= x <= fx + fw and fy <= y <= fy + fh:
            fight_click_count += 1
            print(f"[COUNT] Fight clicked: {fight_click_count}x")

# Start mouse listener
listener = mouse.Listener(on_click=on_click)
listener.start()

try:
    while True:
        # Capture screen
        screen = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= THRESHOLD:
            x, y = max_loc
            fight_box = (x, y, w, h)
        else:
            fight_box = None

        time.sleep(SCAN_DELAY)

except KeyboardInterrupt:
    print("\n[EXIT] Program dihentikan")
    print(f"[RESULT] Total Fight click: {fight_click_count}x")
