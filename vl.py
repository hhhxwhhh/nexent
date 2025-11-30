import os
# 设置环境变量来解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 可选：关闭TensorFlow的oneDNN优化来减少警告信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
from io import BytesIO

# 本地模型路径 - 需要先下载模型到此路径
MODEL_PATH = "./Qwen2.5-VL-7B-Instruct"

# 手动实现 process_vision_info 函数
def process_vision_info(conversations, return_video_kwargs=False):
    """Extract vision info from conversations."""
    def extract_vision_info(conversations):
        vision_infos = []
        if isinstance(conversations, list) and len(conversations) > 0:
            if isinstance(conversations[0], dict):
                # Single conversation
                messages = conversations
            elif isinstance(conversations[0], list):
                # Batch of conversations
                messages = conversations[0]
            else:
                messages = conversations
                
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and ("image" in item or "image_url" in item or "video" in item):
                                vision_infos.append(item)
        return vision_infos
    
    def fetch_image(vision_info):
        if "image" in vision_info:
            image_source = vision_info["image"]
        elif "image_url" in vision_info:
            image_source = vision_info["image_url"]
        else:
            return None
            
        if isinstance(image_source, str):
            if image_source.startswith("http"):
                # URL image
                response = requests.get(image_source)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Local file
                image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
            
        return image
    
    vision_infos = extract_vision_info(conversations)
    
    image_inputs = []
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            try:
                image_inputs.append(fetch_image(vision_info))
            except Exception as e:
                print(f"Error fetching image: {e}")
        
    if len(image_inputs) == 0:
        image_inputs = None
        
    if return_video_kwargs:
        return image_inputs, None, {}
    return image_inputs, None

def download_model_with_modelscope():
    """
    使用ModelScope下载Qwen2.5-VL模型到本地
    这是推荐的方法，因为ModelScope是国内阿里云提供的镜像站，下载速度更快
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        
        print("开始从ModelScope下载Qwen2.5-VL-7B-Instruct模型...")
        model_dir = snapshot_download(
            'Qwen/Qwen2.5-VL-7B-Instruct',
            local_dir=MODEL_PATH,
            revision='master'
        )
        print(f"模型已成功下载到: {model_dir}")
        return model_dir
    except ImportError:
        print("未安装ModelScope，请先运行: pip install modelscope")
        return None
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        return None

def download_model_with_hf_mirror():
    """
    使用HuggingFace镜像站(hf-mirror.com)下载Qwen2.5-VL模型
    这是另一种国内加速下载的方法
    """
    import os
    try:
        # 设置环境变量使用国内镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        from huggingface_hub import snapshot_download
        
        print("开始从HF镜像站下载Qwen2.5-VL-7B-Instruct模型...")
        model_dir = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=MODEL_PATH,
            resume_download=True
        )
        print(f"模型已成功下载到: {model_dir}")
        return model_dir
    except ImportError:
        print("未安装huggingface_hub，请先运行: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        return None

def download_model_with_git_lfs():
    """
    使用git-lfs从HF镜像站克隆模型
    这种方法适合网络不稳定需要断点续传的情况
    """
    import subprocess
    import os
    
    try:
        # 确保安装了git-lfs
        subprocess.run(["git", "lfs", "install"], check=True)
        
        # 克隆模型仓库
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        print("开始使用git-lfs从HF镜像站克隆模型...")
        subprocess.run([
            "git", "clone", "https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct",
            model_path
        ], check=True)
        
        # 拉取大文件
        subprocess.run(["git", "lfs", "pull"], cwd=model_path, check=True)
        
        print(f"模型已成功下载到: {model_path}")
        return model_path
    except subprocess.CalledProcessError as e:
        print(f"使用git-lfs下载模型时出现错误: {e}")
        return None
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        return None

def main():
    # 检查模型是否存在
    import os
    if not os.path.exists(MODEL_PATH):
        print("模型不存在，请先下载模型！")
        return
    
    # 初始化模型和处理器
    print("正在加载模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 准备输入数据
    image_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                },
                {"type": "text", "text": "Please provide a detailed description of this image"},
            ],
        },
    ]

    messages = image_messages

    # 处理图像
    image_inputs, _ = process_vision_info(messages)

    # 应用聊天模板
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 准备模型输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 生成输出
    print("正在生成描述...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("生成结果:")
    print(output_text[0])

if __name__ == "__main__":
    # 首先检查模型是否已存在
    import os
    if not os.path.exists(MODEL_PATH):
        print("模型不存在，开始下载...")
        print("请选择下载方式:")
        print("1. 使用ModelScope下载（推荐）")
        print("2. 使用HF镜像站下载")
        print("3. 使用git-lfs下载")
        
        choice = input("请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            download_model_with_modelscope()
        elif choice == "2":
            download_model_with_hf_mirror()
        elif choice == "3":
            download_model_with_git_lfs()
        else:
            print("无效选项，退出程序")
            exit(1)
    
    # 运行推理
    main()