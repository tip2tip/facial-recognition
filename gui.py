import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QInputDialog, QTextEdit, QMessageBox, QDialog, QFrame
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2
from PIL import Image
from main import FaceRecognitionSystem

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.face_system = FaceRecognitionSystem()
        self.init_ui()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.auto_save_faces)
        self.autosave_timer.start(60000)

        # 设置全局样式表
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5f9e;
            }
            QListWidget {
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
        """)

    def init_ui(self):
        self.setWindowTitle('人脸识别系统')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #f5f5f5;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 左侧视频显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 创建视频显示框架
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #dcdcdc;
                border-radius: 10px;
            }
        """)
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: none;")
        video_layout.addWidget(self.video_label)
        left_layout.addWidget(video_frame)

        layout.addWidget(left_widget)

        # 右侧控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        # 控制按钮组
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        buttons_layout = QVBoxLayout(buttons_frame)
        buttons_layout.setSpacing(10)

        self.start_button = QPushButton('开始识别')
        self.register_button = QPushButton('注册新人脸')
        self.delete_button = QPushButton('删除人脸')

        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.register_button)
        buttons_layout.addWidget(self.delete_button)

        right_layout.addWidget(buttons_frame)

        # 已注册人脸列表区域
        list_frame = QFrame()
        list_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        list_layout = QVBoxLayout(list_frame)
        list_label = QLabel('已注册的人脸：')
        list_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.face_list = QListWidget()
        list_layout.addWidget(list_label)
        list_layout.addWidget(self.face_list)

        right_layout.addWidget(list_frame)

        # 识别信息显示区域
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_label = QLabel('识别信息：')
        info_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(200)
        info_layout.addWidget(info_label)
        info_layout.addWidget(self.info_text)

        right_layout.addWidget(info_frame)

        # 退出按钮
        self.exit_button = QPushButton('退出程序')
        self.exit_button.setStyleSheet("""
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            min-width: 120px;
        """)
        right_layout.addWidget(self.exit_button)

        layout.addWidget(right_widget)

        # 连接按钮信号
        self.start_button.clicked.connect(self.toggle_recognition)
        self.register_button.clicked.connect(self.register_face)
        self.delete_button.clicked.connect(self.delete_face)
        self.exit_button.clicked.connect(self.close)

        self.update_face_list()

    def delete_face(self):
        # 获取选中的人脸
        current_item = self.face_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, '警告', '请先选择要删除的人脸')
            return

        name = current_item.text()
        # 确认删除
        reply = QMessageBox.question(self, '确认删除',
                                   f'确定要删除{name}的人脸数据吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 从known_faces中删除
            if name in self.face_system.known_faces:
                del self.face_system.known_faces[name]
                # 保存更新后的人脸数据
                self.face_system.save_faces()
                # 更新列表显示
                self.update_face_list()
                QMessageBox.information(self, '成功', f'已删除{name}的人脸数据')
            else:
                QMessageBox.warning(self, '错误', f'未找到{name}的人脸数据')

    def update_face_list(self):
        self.face_list.clear()
        for name in self.face_system.known_faces.keys():
            self.face_list.addItem(name)

    def toggle_recognition(self):
        if self.timer.isActive():
            self.stop_recognition()
        else:
            self.start_recognition()

    def start_recognition(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.start_button.setText('停止识别')
        self.timer.start(30)

    def stop_recognition(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_button.setText('开始识别')
        self.video_label.clear()

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 处理帧并进行人脸识别
                frame, info_text = self.face_system.process_frame(frame)
                # 更新识别信息显示
                self.info_text.setText(info_text)
                # 转换图像格式以在Qt中显示
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # 调整图像大小以适应标签
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)

    def register_face(self):
        name, ok = QInputDialog.getText(self, '注册新人脸', '请输入姓名：')
        if ok and name:
            # 显示操作提示
            msg = QMessageBox()
            msg.setWindowTitle('操作提示')
            msg.setText('请面对摄像头，按空格键拍照，按ESC键退出')
            msg.exec_()

            # 创建一个新窗口用于显示摄像头画面
            capture_window = QDialog(self)
            capture_window.setWindowTitle('人脸注册')
            capture_window.setGeometry(200, 200, 800, 600)
            layout = QVBoxLayout(capture_window)
            video_label = QLabel()
            video_label.setMinimumSize(640, 480)
            layout.addWidget(video_label)

            # 初始化摄像头
            self.cap = cv2.VideoCapture(0)
            timer = QTimer()

            def update_frame():
                ret, frame = self.cap.read()
                if ret:
                    # 实时检测人脸
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    faces, _ = self.face_system.mtcnn.detect(frame_pil)

                    # 在画面中标记检测到的人脸
                    if faces is not None:
                        for box in faces:
                            box = box.astype(int)
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                    # 显示实时画面
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    video_label.setPixmap(scaled_pixmap)

            def keyPressEvent(event):
                if event.key() == Qt.Key_Space:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        faces, _ = self.face_system.mtcnn.detect(frame_pil)
                        if faces is not None and len(faces) > 0:
                            if self.face_system.add_face(name, frame):
                                self.update_face_list()
                                QMessageBox.information(capture_window, '注册成功', f'已成功注册{name}的人脸数据')
                                capture_window.accept()
                            else:
                                QMessageBox.warning(capture_window, '注册失败', '人脸注册失败，请重试')
                        else:
                            QMessageBox.warning(capture_window, '注册失败', '未检测到人脸，请调整位置后重试')
                elif event.key() == Qt.Key_Escape:
                    capture_window.reject()

            # 设置键盘事件处理
            capture_window.keyPressEvent = keyPressEvent

            # 启动定时器更新画面
            timer.timeout.connect(update_frame)
            timer.start(30)

            # 显示窗口并等待结果
            result = capture_window.exec_()

            # 清理资源
            timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.video_label.clear()

    def closeEvent(self, event):
        # 停止所有定时器
        self.timer.stop()
        self.autosave_timer.stop()
        # 停止摄像头
        if self.cap is not None:
            self.cap.release()
        # 保存人脸数据
        try:
            self.face_system.save_faces()
            print('人脸数据已保存')
        except Exception as e:
            print('保存人脸数据时出错：', str(e))
        event.accept()

    def auto_save_faces(self):
        try:
            self.face_system.save_faces()
            print('人脸数据已自动保存')
        except Exception as e:
            print('自动保存人脸数据时出错：', str(e))

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()