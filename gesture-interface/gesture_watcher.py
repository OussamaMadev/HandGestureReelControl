import time
import os

class GestureWatcher:
    def __init__(self, file_path="gesture.txt"):
        self.file_path = file_path
        self.last_gesture = ""

    def read_gesture(self):
        time.sleep(1)  # Check every second
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                gesture = f.read().strip()
                if gesture != self.last_gesture:
                    self.last_gesture = gesture
                    return gesture
        return ""
