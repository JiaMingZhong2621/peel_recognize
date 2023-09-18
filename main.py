
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import cv2

from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.capnums import Camera
from dialog.rtsp_win import Window

from engine.model.logit import LogitHead
from engine.config import parser
from engine.model.head import make_classifier_head
import torch.nn as nn
from engine import clip
from engine.clip import partial_model
class_names = ['十五年','五年','一年','十年','二十年']


def make_classifier_head(classifier_head,
                         clip_encoder,
                         bias=False):
    if clip_encoder == 'ViT-B/16':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024

    linear_head = nn.Linear(in_features, 5, bias=bias)

    if classifier_head == 'linear':
        head = linear_head
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, in_features
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)
    send_year = pyqtSignal(str)
    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = 'pt/best_model.pt'           # 设置权重
        self.current_weight = 'pt/best_model.pt'    # 当前权重
        self.source = '0'                       # 视频源
        self.conf_thres = 0.25                  # 置信度
        self.jump_out = False                   # 跳出循环
        self.is_continue = True                 # 继续/暂停
        self.percent_length = 1000              # 进度条
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ
        self.statistic_dic = {name: 0 for name in class_names}
        args = parser.parse_args()

        head, in_features = make_classifier_head(
            args.classifier_head,
            args.clip_encoder
        )
        self.logit_head = LogitHead(
            head,
            logit_scale=4.60517,
        ).eval()
        self.clip_model, _ = clip.load(args.clip_encoder, jit=False, device='cpu')
        self.clip_model.float()
        self.clip_model.eval()
        self.image_encoder = partial_model.get_image_encoder(
            args.clip_encoder,
            args.image_layer_idx,
            self.clip_model
        )
        result_dict = torch.load('pt/best_model.pth', map_location=torch.device('cpu'))

        self.logit_head.load_state_dict(result_dict["logit_head"])
    @torch.no_grad()
    def run(self,
            imgsz=224,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=1,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            half &= device != 'cpu'  # half precision only supported on CUDA
            print(device)
            # Load model

            self.image_encoder.feature_extractor.eval()
            if half:
                self.image_encoder.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=224)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=224)
            count = 0
            # 跳帧检测
            dataset = iter(dataset)
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('停止')
                    break
                # 临时更换模型
                if self.current_weight != self.weights:
                    # Load model
                    result_dict = torch.load('pt/best_model.pth',map_location=torch.device('cpu'))
                    self.image_encoder.feature_extractor.eval()
                    self.logit_head.load_state_dict(result_dict["logit_head"])
                    if half:
                        self.image_encoder.half()  # to FP16
                    self.current_weight = self.weights
                # 暂停开关
                if self.is_continue:
                    path, img, img0= next(dataset)
                    count += 1
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    image_feature = self.image_encoder.feature_extractor(img)
                    pred = self.logit_head(image_feature)
                    print(pred)
                    pred = torch.argmax(pred, dim=1).cpu().numpy()
                    # self.send_raw.emit(img0 if isinstance(img0, np.ndarray) else img0[0])
                    self.send_img.emit(np.array(img0))
                    self.send_year.emit(class_names[int(pred)])

                    self.statistic_dic[class_names[int(pred)]] += 1
                    self.send_statistic.emit(self.statistic_dic)

                    self.send_percent.emit(0)
                    self.send_msg.emit('检测结束')


        except Exception as e:
            self.send_msg.emit('%s' % e)

class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)  # 继承自Ui_mainWindow中的setupUi方法
        self.m_flag = False
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type           # 权重
        self.det_thread.source = '0'                                    # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_year.connect(lambda x:self.label_year.setText(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        # self.comboBox.currentTextChanged.connect(lambda x: self.statistic_msg('模型切换为%s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('pt')
        pt_list = [file for file in pt_list if file.endswith('.pth')]
        pt_list.sort(key=lambda x: os.path.getsize('pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def checkrate(self):
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在检测摄像头设备', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                # 使用exec函数执行python代码
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            conf = 0.33
            rate = 10
            check = 0
            new_config = {"conf": 0.33,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            conf = config['conf']
            rate = config['rate']
            check = config['check']
        self.confSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    def open_file(self):
        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'config/setting.json'
        config = dict()
        config['conf'] = self.confSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='请稍等，正在关闭程序。。。', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())

