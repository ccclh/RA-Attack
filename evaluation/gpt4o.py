import os
import sys
import json
import base64
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from pprint import pprint

import os
import sys
import json
import base64
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from pprint import pprint
import argparse
import json
import os
import time
from openai import OpenAI

print(os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qwen_vl_utils import process_vision_info
from utils import read_mm_safebench, ImageProcessor ,process_hades_data, process_data_dataset, build_ra_attack_prompt

try:
    from utils import build_ra_attack_prompt
except ImportError:
    print("错误: 无法从 'utils.py' 导入 'build_ra_attack_prompt'。")
    print("请确保 'utils.py' 文件存在，并包含所需的函数。")
    sys.exit(1)

# --- 2. 配置 ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
# --- 3. 核心功能函数 ---

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        print(f"错误: 无法读取图片文件 {image_path}。错误信息: {e}")
        return None

def generate_gpt4o_response(query, image_path, model_name):
    """使用OpenAI API调用GPT-4o并获取响应"""
    if not OPENAI_API_KEY or "在此处填入" in OPENAI_API_KEY:
        return {"error": "OpenAI API密钥未配置。", "status": "error"}

    client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)
    base64_image = encode_image(image_path)
    if base64_image is None:
        return {"error": f"图片编码失败: {image_path}", "status": "error"}

    messages = [{"role": "user", "content": [{"type": "text", "text": query}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}]}]
    params = {"model": model_name, "messages": messages,"max_completion_tokens": 4096}

    try:
        response = client.chat.completions.create(**params, timeout=60)  
        return {"content": response.choices[0].message.content, "status": "success"}
    
    except Exception as e:
        return {"error": f"API调用失败: {e}", "status": "error"}

def load_dataset(base_dir):
    """
    专门用于加载 advbench 数据结构的函数。
    """
    instruction_path = os.path.join(base_dir, 'instruction', f"{args.dataset}.json")
    mindmap_dir = os.path.join(base_dir, 'mindmap')
    
    IMAGE_ID_KEY = 'mind_map_id'
    IMAGE_EXTENSION = '.png'

    if not os.path.exists(instruction_path):
        raise FileNotFoundError(f"错误: 指令文件未找到: {instruction_path}")
    if not os.path.isdir(mindmap_dir):
        raise NotADirectoryError(f"错误: 思维导图目录不存在: {mindmap_dir}")

    with open(instruction_path, 'r', encoding='utf-8') as f:
        all_instructions = json.load(f)

    processed_dataset = []
    for item in all_instructions:
        if IMAGE_ID_KEY not in item:
            continue
        
        image_id = item[IMAGE_ID_KEY]
        image_filename = image_id + IMAGE_EXTENSION
        
        full_image_path = os.path.join(mindmap_dir, image_id, image_filename)
        if os.path.exists(full_image_path):
            item['mindmap_image'] = full_image_path
            processed_dataset.append(item)
        else:
            print(f"警告: 找不到对应的图片 '{full_image_path}'，已跳过。")
            
    print(f"从 '{instruction_path}' 成功加载并处理了 {len(processed_dataset)} 条数据。")
    return processed_dataset

# --- 4. 主逻辑 ---

def main(args):
    """主执行函数"""
    try:

        dataset = load_dataset(args.input_dir)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return
    
    if args.ids and args.range:
        print("错误: --ids 和 --range 参数不能同时使用。请只选择一种样本筛选方式。")
        sys.exit(1)

    if args.ids:
        print(f"检测到 --ids 参数，将只处理ID为 {args.ids} 的样本。")
        selected_ids = set(args.ids)
        dataset = [item for item in dataset if item.get('id') in selected_ids]
    elif args.range:
        try:
            start_str, end_str = args.range.split('-')
            start_id = int(start_str)
            end_id = int(end_str)
            if start_id > end_id:
                raise ValueError("起始ID不能大于终止ID")

            print(f"检测到 --range 参数，将处理ID从 {start_id} 到 {end_id} 的样本。")
            selected_ids = set(range(start_id, end_id + 1))
            dataset = [item for item in dataset if item.get('id') in selected_ids]
        except ValueError as e:
            print(f"错误: --range 参数格式不正确。应为 '起始ID-终止ID'，例如 '1-3'。详细错误: {e}")
            sys.exit(1)
    
    if not dataset:
        print("警告: 根据您提供的ID或范围，没有找到任何可处理的样本。脚本将提前结束。")
        return
    # -------------------------
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.model_name.replace('/', '_')}.jsonl")

    with open(output_file, "w", encoding="utf-8") as fout:
        for item in tqdm(dataset, desc=f"在 {args.dataset} 上处理查询"):
            
            if args.attack_type == 'CA':
                query = build_ra_attack_prompt(item, include_mindmap=True) 
                image_path = item.get("mindmap_image", "")
            elif args.attack_type == 'ori':
                query = item.get("original_instruction", "")
                image_path = item.get("mindmap_image", "")
            else:
                query = item.get("original_instruction", "")
                image_path = item.get("mindmap_image", "")
             
            if not image_path:
                print(f"警告: ID 为 {item.get('id')} 的项图片路径为空，已跳过。")
                continue
            
            response_data = generate_gpt4o_response(query, image_path, args.model_name)
         
            
            result = {
                "scenario": item.get("scenario"),
                "id": item.get("id"),
                "original_instruction": item.get("original_instruction"),
                "professional_harmful_intent": item.get("professional_harmful_intent"),
                "category": item.get("category"),
                "role": item.get("role"),
                "mind_map_id": item.get("mind_map_id"),
                "mind_map_theme": item.get("mind_map_theme"),
                "query": query,
                "image": item.get("mindmap_image", ""),
                "response": response_data.get("content", ""),
                "error": response_data.get("error")
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

# --- 5. 启动入口 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用GPT-4o API对AdvBench数据集执行攻击")
    
    parser.add_argument('--attack_type', type=str, default='CA', choices=['CA', 'ori'],
                        help="要执行的攻击类型")
    
    parser.add_argument('--model_name', type=str, default='gpt-4o',choices=['gpt-4o', 'o4-mini','gpt-5-mini-2025-08-07'],
                        help='要使用的OpenAI模型名称')
    
    parser.add_argument('--dataset', default="advbench",
                        help='数据集名称，主要用于命名输出目录')
    
    parser.add_argument('--input_dir', default="./Advbench",
                        help='包含 instruction 和 mindmap 文件夹的数据根目录')
    
    parser.add_argument('--output_dir', default="./data_RA_response",
                        help='保存输出JSON文件的根目录')
    parser.add_argument('--ids', type=int, nargs='+', help='要处理的非连续样本ID列表 (例如: --ids 5 12 23)')
    parser.add_argument('--range', type=str, help='要处理的连续样本ID范围，格式为 "起始-结束" (例如: --range 1-3)')
   
    args = parser.parse_args()
    
    print("脚本启动，使用以下参数:")
    pprint(vars(args))
    
    main(args)
    
    print(f"\n脚本执行完毕。结果已保存到 '{os.path.join(args.output_dir, args.dataset)}' 目录中。")
