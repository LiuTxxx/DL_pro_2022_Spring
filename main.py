from PyQt5.QtWidgets import *
import threading
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *
import cv2
import os
import utils


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.uploading = None
        self.setWindowTitle('Face Recognition')
        self.resize(1100, 650)
        self.setWindowIcon(QIcon("UI_images/logo.png"))
        self.up_img_name = ""
        self.input_fname = ""
        self.source = ''
        self.video_capture = cv2.VideoCapture(0)
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.init_ui()
        self.set_down()

    def init_ui(self):
        font_v = QFont('楷体', 14)
        generally_font = QFont('楷体', 15)
        img_widget = QWidget()
        img_layout = QVBoxLayout()
        img_f_title = QLabel("The upload image")
        img_f_title.setAlignment(Qt.AlignCenter)
        img_f_title.setFont(QFont('楷体', 18))

        self.img_f_img = QLabel()
        self.img_f_img.setPixmap(QPixmap("UI_images/sustech1.png"))
        self.img_f_img.setAlignment(Qt.AlignCenter)
        self.face_name = QLineEdit()

        img_upload_btn = QPushButton("Upload Face")
        img_start_btn = QPushButton("Start")
        img_upload_btn.clicked.connect(self.up_img)
        img_start_btn.clicked.connect(self.start_up)
        # set style
        img_upload_btn.setFont(generally_font)
        img_start_btn.setFont(generally_font)
        img_upload_btn.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_start_btn.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")

        img_layout.addWidget(img_f_title)
        img_layout.addWidget(self.img_f_img)
        img_layout.addWidget(self.face_name)
        img_layout.addWidget(img_upload_btn)
        img_layout.addWidget(img_start_btn)
        img_widget.setLayout(img_layout)

        # recognition part
        video_widget = QWidget()
        video_layout = QVBoxLayout()

        # lable
        self.video_title2 = QLabel("Recognition")
        self.video_title2.setFont(font_v)
        self.video_title2.setAlignment(Qt.AlignCenter)
        self.video_title2.setFont(font_v)

        # camera button
        self.DisplayLabel = QLabel()
        self.DisplayLabel.setPixmap(QPixmap(""))
        self.open_camera_btn = QPushButton("check camera")
        self.open_camera_btn.setFont(font_v)
        self.open_camera_btn.setStyleSheet("QPushButton{color:white}"
                                           "QPushButton:hover{background-color: rgb(2,110,180);}"
                                           "QPushButton{background-color:rgb(48,124,208)}"
                                           "QPushButton{border:2px}"
                                           "QPushButton{border-radius:5px}"
                                           "QPushButton{padding:5px 5px}"
                                           "QPushButton{margin:5px 5px}")

        # start button
        self.start_rec_btn = QPushButton("Start Recognition (choose video)")
        self.start_rec_btn.setFont(font_v)
        self.start_rec_btn.setStyleSheet("QPushButton{color:white}"
                                         "QPushButton:hover{background-color: rgb(2,110,180);}"
                                         "QPushButton{background-color:rgb(48,124,208)}"
                                         "QPushButton{border:2px}"
                                         "QPushButton{border-radius:5px}"
                                         "QPushButton{padding:5px 5px}"
                                         "QPushButton{margin:5px 5px}")

        # stop button
        self.stop_rec_btn = QPushButton("Stop Recognition")
        self.stop_rec_btn.setFont(font_v)
        self.stop_rec_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")

        # connect function
        self.open_camera_btn.clicked.connect(self.open_local)
        self.start_rec_btn.clicked.connect(self.open)
        self.stop_rec_btn.clicked.connect(self.close)

        video_layout.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_title2)
        video_layout.addWidget(self.DisplayLabel)
        self.DisplayLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.open_camera_btn)
        video_layout.addWidget(self.start_rec_btn)
        video_layout.addWidget(self.stop_rec_btn)
        video_widget.setLayout(video_layout)

        # add each tab
        self.addTab(img_widget, "Upload")
        self.addTab(video_widget, 'Recognition')
        self.setTabIcon(0, QIcon('UI_images/图片.png'))
        self.setTabIcon(1, QIcon('UI_images/直播.png'))

    def up_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'choose file', '', 'Image files(*.jpg , *.png)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            # img = cv2.imread(img_name)
            # self.uploading = img
            # self.img_f_img.setPixmap(QPixmap(img_name))
            src_img = cv2.imread(img_name)
            self.uploading = src_img
            src_img_height = src_img.shape[0]
            src_img_width = src_img.shape[1]
            target_img_height = 400
            ratio = target_img_height / src_img_height
            target_img_width = int(src_img_width * ratio)
            target_img = cv2.resize(src_img, (target_img_width, target_img_height))
            cv2.imwrite("UI_images/tmp/temp.jpg", target_img)
            self.img_f_img.setPixmap(QPixmap("UI_images/tmp/temp.jpg"))

    def start_up(self):
        face_name = self.face_name.text()
        if face_name == "":
            QMessageBox.information(self, "Error", "Name cannot be NULL")
        elif utils.add_face(self.uploading, face_name):
            QMessageBox.information(self, "Succeed", "Add face success")
        else:
            QMessageBox.information(self, "Fail", "Too many or no face detected")

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'Exit',
                                     "Do you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    def open(self):
        mp4_fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4')
        if mp4_fileName:
            self.source = mp4_fileName
            self.video_capture = cv2.VideoCapture(self.source)
            if self.video_capture.isOpened():
                th = threading.Thread(target=self.display_video)
                th.start()

    def open_local(self):
        mp4_filename = 0
        self.source = mp4_filename
        self.video_capture = cv2.VideoCapture(self.source)
        if self.video_capture.isOpened():
            th = threading.Thread(target=self.display_video)
            th.start()
        else:
            QMessageBox.information(self, "Error", "Camera not ready")


    def close(self):
        self.stopEvent.set()
        self.set_down()

    def display_video(self):
        # disable start btn
        utils.update_face()
        self.open_camera_btn.setEnabled(False)
        self.start_rec_btn.setEnabled(False)
        self.stop_rec_btn.setEnabled(True)
        process_this_frame = True
        while True:
            ret, frame = self.video_capture.read()
            if ret:
                frame = utils.rec_frame(frame)
                frame_height = frame.shape[0]
                frame_width = frame.shape[1]
                frame_scale = 500 / frame_height
                frame_resize = cv2.resize(frame, (int(frame_width * frame_scale), int(frame_height * frame_scale)))
                cv2.imwrite("UI_images/tmp.jpg", frame_resize)
                self.DisplayLabel.setPixmap(QPixmap("UI_images/tmp.jpg"))
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.DisplayLabel.clear()
                self.stop_rec_btn.setEnabled(False)
                self.open_camera_btn.setEnabled(True)
                self.start_rec_btn.setEnabled(True)
                self.set_down()
                break
        self.start_rec_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)
        self.set_down()

    def set_down(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.DisplayLabel.setPixmap(QPixmap("UI_images/sustech1.png"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
