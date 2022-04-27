import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os


class CaptureFace:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=self.device)

    def capture_face(self, dir_dataset, dir_images):
        count = 50
        usr_name = input("Input ur name: ")
        USR_PATH = os.path.join(dir_images, usr_name)
        leap = 1

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while cap.isOpened() and count:
            isSuccess, frame = cap.read()
            if self.mtcnn(frame) is not None and leap % 2:
                path = str(USR_PATH + '/{}.jpg'.format(
                    str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + str(count)))
                face_img = self.mtcnn(frame, save_path=path)
                count -= 1
            leap += 1
            cv2.imshow('Face Capturing', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
