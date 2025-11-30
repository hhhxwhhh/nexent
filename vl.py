import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# 本地模型路径 - 需要先下载模型到此路径
MODEL_PATH = "./Qwen2.5-VL-7B-Instruct"

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
    # 初始化模型和处理器
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    sampling_params = SamplingParams(
        max_tokens=512
    )

    image_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": "Please provide a detailed description of this image"},
            ],
        },
    ]

    messages = image_messages

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)

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