import cv2


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(2)

        self.is_record = False
        self.out = None
        self.transformed_frame = None

        self.cached_frame = None

    def __del_(self):
        self.cap.release()

    def get_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.cached_frame = frame
            ret, jpeg = cv2.imencode(".jpg", frame)
            return jpeg.tobytes()
        else:
            return None

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            ret, jpeg = cv2.imencode(".jpg", frame)
            self.transformed_frame = jpeg.tobytes()
        else:
            return None

    def get_cached_frame(self):
        return self.cached_frame

    def get_image_frame(self):
        return self.transformed_frame
