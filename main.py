import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import os
import json

class FaceRecognitionSystem:
    def __init__(self):
        # 初始化MTCNN用于人脸检测
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            # 初始化InceptionResnetV1用于特征提取
            model_path = 'model/20180402-114759-vggface2.pt'
            if os.path.exists(model_path):
                # 如果本地模型文件存在，则从本地加载
                self.resnet = InceptionResnetV1(pretrained=None).eval().to(self.device)
                state_dict = torch.load(model_path, weights_only=True)
                # 移除不匹配的权重
                for key in list(state_dict.keys()):
                    if key.startswith('logits'):
                        del state_dict[key]
                self.resnet.load_state_dict(state_dict, strict=False)
            else:
                # 如果本地模型不存在，则尝试从网络下载
                print('正在从网络下载预训练模型...')
                os.makedirs('model', exist_ok=True)
                self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                # 保存模型到本地
                torch.save(self.resnet.state_dict(), model_path)
                print('模型已保存到本地：', model_path)
        except Exception as e:
            print('初始化人脸识别系统时出错：', str(e))
            raise
        # 存储已知人脸的特征
        self.known_faces = {}
        # 创建faces目录用于存储人脸数据
        self.faces_dir = 'faces'
        os.makedirs(self.faces_dir, exist_ok=True)
        # 加载已保存的人脸数据
        self.load_faces()

    def add_face(self, name, image):
        """添加已知人脸到数据库，包含数据增强"""
        # 转换为PIL Image以进行数据增强
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # 数据增强变换
        augmented_images = [
            image_pil,  # 原始图像
            image_pil.transpose(Image.FLIP_LEFT_RIGHT),  # 水平翻转
            image_pil.rotate(5),  # 轻微顺时针旋转
            image_pil.rotate(-5),  # 轻微逆时针旋转
        ]
        
        # 为每个增强后的图像提取特征
        embeddings = []
        for aug_img in augmented_images:
            # 检测人脸
            faces = self.mtcnn(aug_img)
            if faces is not None and len(faces) > 0:
                # 提取特征并添加到列表
                face_embedding = self.resnet(faces[0].unsqueeze(0))
                embeddings.append(face_embedding.detach())
        
        if embeddings:
            # 计算平均特征向量，确保维度为 [1, 512]
            mean_embedding = torch.mean(torch.stack(embeddings), dim=0).unsqueeze(0)
            self.known_faces[name] = mean_embedding
            # 保存人脸数据
            self.save_faces()
            return True
        return False

    def save_faces(self):
        """保存人脸数据到文件"""
        try:
            # 将人脸特征转换为可序列化的格式，确保维度正确
            face_data = {}
            for name, embedding in self.known_faces.items():
                # 确保embedding是二维张量，并移除多余的维度
                if len(embedding.shape) > 2:
                    embedding = embedding.squeeze()
                if len(embedding.shape) == 1:
                    embedding = embedding.unsqueeze(0)
                
                # 如果仍有多个样本，取平均值
                if embedding.shape[0] > 1:
                    embedding = embedding.mean(dim=0).unsqueeze(0)
                
                # 验证特征向量维度
                if embedding.shape != (1, 512):
                    print(f'警告：{name}的特征向量维度不正确 {embedding.shape}，已跳过保存')
                    continue
                
                # 保存特征向量
                face_data[name] = {
                    'embedding': embedding.cpu().detach().numpy().tolist()[0],  # 只保存一维数组
                    'shape': [1, 512]  # 记录标准维度
                }
            
            # 保存到JSON文件
            face_file = os.path.join(self.faces_dir, 'face_data.json')
            with open(face_file, 'w') as f:
                json.dump(face_data, f)
            print(f'成功保存 {len(face_data)} 个人脸数据')
        except Exception as e:
            print('保存人脸数据时出错：', str(e))

    def load_faces(self):
        """从文件加载人脸数据"""
        try:
            face_file = os.path.join(self.faces_dir, 'face_data.json')
            if os.path.exists(face_file):
                with open(face_file, 'r') as f:
                    face_data = json.load(f)
                
                loaded_count = 0
                for name, data in face_data.items():
                    try:
                        # 获取特征向量并转换为张量
                        embedding_list = data['embedding']
                        embedding_tensor = torch.tensor(embedding_list).to(self.device)
                        
                        # 确保维度正确 [1, 512]
                        if len(embedding_tensor.shape) > 2:
                            embedding_tensor = embedding_tensor.squeeze()
                        if len(embedding_tensor.shape) == 1:
                            embedding_tensor = embedding_tensor.unsqueeze(0)
                        
                        # 如果仍有多个样本，取平均值
                        if embedding_tensor.shape[0] > 1:
                            embedding_tensor = embedding_tensor.mean(dim=0).unsqueeze(0)
                        
                        # 验证维度
                        if embedding_tensor.shape != (1, 512):
                            print(f'警告：{name}的特征向量维度不正确 {embedding_tensor.shape}，已跳过加载')
                            continue
                        
                        self.known_faces[name] = embedding_tensor
                        loaded_count += 1
                    except Exception as e:
                        print(f'加载{name}的人脸数据时出错：{str(e)}')
                        continue
                        
                print(f'已成功加载 {loaded_count} 个人脸数据')
        except Exception as e:
            print('加载人脸数据时出错：', str(e))

    def recognize_face(self, face):
        """识别人脸"""
        if face is None:
            return None, float('inf')
        # 提取特征
        with torch.no_grad():
            face_embedding = self.resnet(face.unsqueeze(0))
        # 计算与已知人脸的余弦相似度
        max_similarity = -1
        best_match = None
        for name, known_embedding in self.known_faces.items():
            # 确保特征向量是二维张量 [1, 512]
            if len(known_embedding.shape) == 1:
                known_embedding = known_embedding.unsqueeze(0)
            if len(face_embedding.shape) == 1:
                face_embedding = face_embedding.unsqueeze(0)
            # 使用余弦相似度，确保维度匹配
            similarity = torch.nn.functional.cosine_similarity(face_embedding, known_embedding, dim=1).mean().item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name
        # 将相似度转换为距离度量（1-相似度），以保持与原有逻辑兼容
        distance = 1 - max_similarity
        return best_match, distance

    def process_frame(self, frame):
        """处理视频帧"""
        # 转换为PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # 检测人脸
        faces, _ = self.mtcnn.detect(frame_pil)
        info_text = ""
        if faces is not None:
            info_text += f"检测到 {len(faces)} 个人脸\n"
            # 获取人脸图像
            faces_tensor = self.mtcnn(frame_pil)
            if faces_tensor is not None:
                for i, face in enumerate(faces_tensor):
                    # 识别人脸
                    name, dist = self.recognize_face(face)
                    
                    # 在图像上绘制边界框和名字
                    box = faces[i].astype(int)
                    threshold = 0.4  # 降低阈值，因为现在使用的是距离度量（1-相似度）
                    if dist < threshold:
                        confidence = ((threshold - dist) / threshold) * 100
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        name_text = f"{name} ({confidence:.1f}%)"
                        cv2.putText(frame, name_text, (box[0], box[1]-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        info_text += f"人脸 {i+1}: 识别为 {name}，置信度: {confidence:.1f}%\n"
                    else:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(frame, "未知", (box[0], box[1]-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        info_text += f"人脸 {i+1}: 未识别出匹配的人脸 (相似度得分过低: {dist:.2f})\n"
        return frame, info_text

    def register_face(self):
        """通过摄像头注册新的人脸"""
        name = input('请输入要注册的人名: ')
        if not name:
            print('名字不能为空')
            return

        cap = cv2.VideoCapture(0)
        registered = False
        print('按空格键拍照，按ESC键退出')

        while not registered:
            ret, frame = cap.read()
            if not ret:
                break

            # 显示实时画面
            cv2.imshow('Register Face', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break
            elif key == 32:  # 空格键拍照
                # 尝试添加人脸
                if self.add_face(name, frame):
                    print(f'成功注册人脸: {name}')
                    registered = True
                else:
                    print('未检测到人脸，请重试')

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """运行人脸识别系统"""
        while True:
            print('\n1. 开始人脸识别')
            print('2. 注册新的人脸')
            print('3. 查看已注册的人脸')
            print('4. 退出')
            choice = input('请选择操作: ')

            if choice == '1':
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 处理帧
                    frame = self.process_frame(frame)

                    # 显示结果
                    cv2.imshow('Face Recognition', frame)

                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

            elif choice == '2':
                self.register_face()

            elif choice == '3':
                if not self.known_faces:
                    print('暂无已注册的人脸')
                else:
                    print('已注册的人脸：')
                    for name in self.known_faces.keys():
                        print(f'- {name}')

            elif choice == '4':
                # 退出前保存人脸数据
                self.save_faces()
                break

            else:
                print('无效的选择，请重试')

if __name__ == '__main__':
    # 创建人脸识别系统实例
    face_system = FaceRecognitionSystem()
    # 运行系统
    face_system.run()