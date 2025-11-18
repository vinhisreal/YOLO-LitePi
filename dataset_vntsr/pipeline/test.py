from picamera2 import Picamera2
import cv2

picam = Picamera2()
picam.configure(picam.create_preview_configuration({"format": "RGB888"}))
picam.start()

while True:
    frame = picam.capture_array()
    cv2.imshow("Pi Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

picam.stop()
cv2.destroyAllWindows()
