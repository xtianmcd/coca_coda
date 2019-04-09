# #copied from https://github.com/yushulx/web-camera-recorder/blob/master/camera.py
# import cv2
# import threading
#
# class RecordingThread (threading.Thread):
#     def __init__(self, name, camera):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.isRunning = True
#
#         self.cap = camera
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))
#
#     def run(self):
#         while self.isRunning:
#             ret, frame = self.cap.read()
#             if ret:
#                 self.out.write(frame)
#
#         self.out.release()
#
#     def stop(self):
#         self.isRunning = False
#
#     def __del__(self):
#         self.out.release()
#
# class VideoCamera(object):
#     def __init__(self):
#         # Open a camera
#         self.cap = cv2.VideoCapture(2)
#
#         # Initialize video recording environment
#         self.is_record = False
#         self.out = None
#
#         # Thread for recording
#         self.recordingThread = None
#
#     def __del__(self):
#         self.cap.release()
#
#     def get_frame(self):
#         ret, frame = self.cap.read()
#
#         if ret:
#             ret, jpeg = cv2.imencode('.jpg', frame)
#
#
#             return jpeg.tobytes()
#
#         else:
#             return None
#
#     def start_record(self):
#         self.is_record = True
#         self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
#         self.recordingThread.start()
#
#     def stop_record(self):
#         self.is_record = False
#
#         if self.recordingThread != None:
#             self.recordingThread.stop()

import cv2
# copied from https://github.com/log0/video_streaming_with_flask_example/blob/master/camera.py

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        cv2.namedWindow("show")
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        # cv2.imshow("show", jpeg)
        # if not success:
        #     # break
        # k = cv2.waitKey(1)
        #
        # if k%256 == 27:
        #     # ESC pressed
        #     print("Escape hit, closing...")
        #     break
        # elif k%256 == 32:
        #     # SPACE pressed
        #     img_name = "opencv_frame_{}.png".format(img_counter)
        #     cv2.imwrite(img_name, frame)
        #     print("{} written!".format(img_name))
        #     img_counter += 1
        #
        # # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # # so we must encode it into JPEG in order to correctly display the
        # # video stream.
        #
        # self.video.release()
        # cv2.destroyAllWindows()

        return jpeg.tobytes()
