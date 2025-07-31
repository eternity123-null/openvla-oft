import base64
import numpy as np
import cv2
import requests
import time
from typing import Dict, Any, List

class VLAClient:
    """
    一个用于与VLA服务器交互的客户端。
    它能生成模拟的观测数据，并向服务器发送请求以获取动作。
    """
    def __init__(self, vla_server_url: str):
        """
        初始化客户端。
        :param vla_server_url: VLA服务器的URL，例如 'http://localhost:5000/get_action'
        """
        self.vla_server_url = vla_server_url
        print(f"VLA客户端已初始化，将连接到: {self.vla_server_url}")

    def img2b64(self, img: np.ndarray) -> str:
        """将图像(numpy.ndarray)转换为base64编码的字符串"""
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def decode_numpy(self, serialized: Dict[str, str]) -> np.ndarray:
        """
        从服务器返回的特定格式字典中解码numpy数组。
        :param serialized: 包含'__numpy__', 'dtype', 'shape'键的字典
        """
        # 检查是否是预期的格式
        if '__numpy__' not in serialized or 'dtype' not in serialized or 'shape' not in serialized:
            raise ValueError("字典格式错误，无法解码Numpy数组。")
            
        data = base64.b64decode(serialized['__numpy__'])
        return np.frombuffer(data, dtype=np.dtype(serialized['dtype'])).reshape(serialized['shape'])

    def get_fake_observation(self, instruction: str) -> Dict[str, Any]:
        """
        根据给定的指令，生成一个模拟的（假的）观测数据字典。
        这些数据将作为发送给VLA服务器的请求体。
        """
        # 生成随机图像数据
        full_img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
        left_img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)

        # 生成随机的状态向量
        # 注意：原始代码中的state构造有索引问题，这里进行了修正和简化
        # 假设状态向量总共有55个元素
        state = np.random.uniform(low=-1, high=1, size=22).astype(np.float32)
        
        # # 假设state的拼接规则如下 (这是一个示例)
        # gripper_pos = mock_full_state[0:2]   # 2
        # arm_pos = mock_full_state[2:16]      # 14
        # base_pos = np.array([0, 0, 0, 0], dtype=np.float32) # 4
        # waist_pos = mock_full_state[16:18]   # 2
    
        # state = np.concatenate((gripper_pos, arm_pos, base_pos, waist_pos))

        return {
            "full_image": self.img2b64(full_img),
            "left_wrist_image": self.img2b64(left_img),
            "right_wrist_image": self.img2b64(right_img),
            "state": state.tolist(),
            "instruction": instruction,
        }

    def get_action_from_server(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """将观测数据发送到VLA服务器，并获取返回的动作"""
        try:
            print("正在向VLA服务器发送请求...")
            response = requests.post(
                self.vla_server_url,
                json=observation,
                timeout=20.0  # 超时时间可以适当延长
            )
            response.raise_for_status()  # 如果状态码不是2xx，则抛出HTTPError
            print(f"VLA服务器响应状态: {response.status_code}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"错误: VLA服务器请求失败: {e}")
            return {}

def main_loop():
    """
    主程序循环。
    不停地获取用户指令，调用VLA服务器，并显示结果。
    """
    # !!!重要!!! 请将此URL替换为您实际的VLA服务器地址
    VLA_SERVER_URL = "http://127.0.0.1:5000/get_action"
    
    client = VLAClient(vla_server_url=VLA_SERVER_URL)

    print("\n--- VLA 客户端已启动 ---")
    instruction = "pour the ice in the right hand into the cup"

    while True:
        try:
            
            observation = client.get_fake_observation(instruction)
            # print("生成的观测数据 (部分):")
            # print(f"  - 指令: {observation['instruction']}")
            # print(f"  - 状态向量 (前5个元素): {observation['state'][:5]}")

            # 3. 从VLA服务器获取动作
            action_response = client.get_action_from_server(observation)

            # 4. 处理并显示结果
            if not action_response:
                print("未能从服务器获取到有效响应。")
                continue
                
            print("\n--- 服务器响应 ---")
            print(f"原始响应内容: {action_response}")
            
            # 假设服务器返回的动作在 'predicted_action' 键中
            if 'predicted_action' in action_response:
                try:
                    predicted_action_encoded = action_response['predicted_action']
                    # 解码numpy数组
                    action_decoded = client.decode_numpy(predicted_action_encoded)
                    print("\n--- 解码后的动作 ---")
                    print(f"类型: {type(action_decoded)}")
                    print(f"形状: {action_decoded.shape}")
                    print(f"内容 (numpy.ndarray):\n{action_decoded}")
                except (ValueError, TypeError, KeyError) as e:
                    print(f"\n解码动作失败: {e}")
            else:
                print("\n服务器响应中未找到 'predicted_action' 键。")

            # 增加一个小的延时，避免循环过快
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n检测到用户中断 (Ctrl+C)，正在退出...")
            break
        except Exception as e:
            print(f"\n主循环发生未知错误: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main_loop()