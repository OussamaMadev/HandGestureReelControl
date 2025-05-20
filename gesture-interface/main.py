import sys
import cv2
import threading
from PyQt5.QtCore import Qt, QTimer, QUrl 
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from gesture_watcher import GestureWatcher


class MainWindow(QWidget):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("Gesture Reels Controller")
        self.setGeometry(100, 100, 1200, 600)

        # Webcam feed label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        # Instagram viewer
        self.web_view = QWebEngineView()
        self.web_view.load(QUrl("https://www.instagram.com/reels/"))

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.web_view, 1)
        self.setLayout(layout)

        # Start webcam
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Start gesture watcher
        self.gesture_thread = threading.Thread(target=self.watch_gestures, daemon=True)
        self.gesture_thread.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            print("Captured a webcam frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)
        else:
            print("Failed to capture webcam frame")


    def watch_gestures(self):
        watcher = GestureWatcher()
        while True:
            try:
                gesture = watcher.read_gesture()
                if not gesture:
                    continue  # skip empty gesture reads
                print(f"Detected gesture: {gesture}")  # debug print
                if gesture == "scroll_up":
                    self.scroll_instagram(-300)
                elif gesture == "scroll_down":
                    self.scroll_instagram(300)
            except Exception as e:
                print(f"Gesture watcher error: {e}")


    def scroll_instagram(self, delta):
        script = f"window.scrollBy(0, {delta});"
        self.web_view.page().runJavaScript(script)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
