import cv2


class Camera:
    def __init__(self, cam_preset_num=5):
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):
        cnt = 0
        devices = []
        for device in range(0, self.cam_preset_num):
            stream = cv2.VideoCapture(device, cv2.CAP_DSHOW)
            # Grabs the next frame from video file or capturing device.
            # true (non-zero) in the case of success.
            grabbed = stream.grab()
            #  close the already opened file or camera
            stream.release()
            if not grabbed:
                continue
            else:
                cnt = cnt + 1
                devices.append(device)
        return cnt, devices


if __name__ == '__main__':
    cam = Camera()
    cam_num, devices = cam.get_cam_num()
    print(cam_num, devices)
