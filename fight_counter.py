import cv2
import numpy as np
import time
from pynput import mouse
import tkinter as tk
import threading
import mss

# ============ CONFIG ============
TEMPLATE_PATH = "fight.png"
THRESHOLD = 0.85
SCAN_DELAY = 0.4
MONITOR_INDEX = 1
# ================================

template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError("fight.png tidak ditemukan")

w, h = template.shape[::-1]

fight_box = None
fight_count = 0
status_text = "WAITING"

# ============ GUI ============
root = tk.Tk()
root.title("Fight Counter")
root.geometry("220x120")
root.attributes("-topmost", True)
root.resizable(False, False)

lbl_title = tk.Label(root, text="Fight Counter", font=("Arial", 12, "bold"))
lbl_title.pack(pady=5)

lbl_count = tk.Label(root, text="Fight: 0x", font=("Arial", 14))
lbl_count.pack()

lbl_status = tk.Label(root, text="Status: WAITING", font=("Arial", 9))
lbl_status.pack(pady=5)

def update_gui():
    lbl_count.config(text=f"Fight: {fight_count}x")
    lbl_status.config(text=f"Status: {status_text}")
    root.after(300, update_gui)

# ============ MOUSE ============
def on_click(x, y, button, pressed):
    global fight_count
    if pressed and fight_box:
        fx, fy, fw, fh = fight_box
        if fx <= x <= fx + fw and fy <= y <= fy + fh:
            fight_count += 1

mouse.Listener(on_click=on_click).start()

# ============ DETECTION LOOP ============
def detection_loop():
    global fight_box, status_text

    sct = mss.mss()  # ✅ penting: buat di thread
    monitor = sct.monitors[MONITOR_INDEX]

    while True:
        screen = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= THRESHOLD:
            fight_box = (*max_loc, w, h)
            status_text = "READY"
        else:
            fight_box = None
            status_text = "WAITING"

        time.sleep(SCAN_DELAY)

threading.Thread(target=detection_loop, daemon=True).start()

update_gui()
root.mainloop()
