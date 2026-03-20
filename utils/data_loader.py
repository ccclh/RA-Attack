import os
import json
from PIL import Image

def load_mindmap_data(data_root):
    """
    加载思维导图攻击数据
    
    参数:
        data_root (str): data目录的根路径
        
    返回:
        dict: 包含所有攻击数据的字典
    """
    result = {}
    
    # 直接处理data/instruction目录下的所有场景
    instruction_dir = os.path.join(data_root, 'instruction')
    if not os.path.exists(instruction_dir):
        print(f'instruction目录不存在: {instruction_dir}')
        return result
    
    # 遍历instruction目录下的所有场景
    for scenario_dir in os.listdir(instruction_dir):
        scenario_path = os.path.join(instruction_dir, scenario_dir)
        if not os.path.isdir(scenario_path):
            continue
            
        # 加载主要攻击数据
        main_data_file = os.path.join(scenario_path, f'{scenario_dir}.json')
        if os.path.exists(main_data_file):
            with open(main_data_file, 'r', encoding='utf-8') as f:
                main_data = json.load(f)
        else:
            main_data = []
        
        # 加载思维导图映射
        map_data_file = os.path.join(scenario_path, f'{scenario_dir}-map.json')
        if os.path.exists(map_data_file):
            with open(map_data_file, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
        else:
            map_data = []
        
        # 加载中文数据（如果存在）
        chinese_data_file = os.path.join(scenario_path, f'{scenario_dir}-c1.json')
        if os.path.exists(chinese_data_file):
            with open(chinese_data_file, 'r', encoding='utf-8') as f:
                chinese_data = json.load(f)
        else:
            chinese_data = []
        
        result[scenario_dir] = {
            "main_data": main_data,
            "map_data": map_data,
            "chinese_data": chinese_data
        }
    
    return result

def process_data_dataset(data_root, language="en"):
    """
    处理data数据集，返回标准化的数据格式
    
    参数:
        data_root (str): data目录的根路径
        language (str): 语言选择，'en'或'zh'
        
    返回:
        list: 标准化的数据列表
    """
    mindmap_data = load_mindmap_data(data_root)
    result = []
    
    for scenario, data in mindmap_data.items():
        # 选择语言版本的数据
        if language == "zh" and data["chinese_data"]:
            source_data = data["chinese_data"]
        else:
            source_data = data["main_data"]
        
        # 创建思维导图ID到主题的映射
        map_dict = {item["mind_map_id"]: item for item in data["map_data"]}
        
        for item in source_data:
            # 构建思维导图图片路径 - 直接从mindmap根目录获取，忽略Violence子目录
            mind_map_id = item.get("mind_map_id", "1")
            mindmap_image_path = os.path.join(
                data_root, 
                "mindmap",  # 直接从mindmap根目录
                mind_map_id, 
                f"{mind_map_id}.png"
            )
            
            # 检查图片文件是否存在
            if not os.path.exists(mindmap_image_path):
                print(f"警告: 思维导图文件不存在: {mindmap_image_path}")
                continue
            
            # 获取思维导图主题信息
            map_info = map_dict.get(mind_map_id, {})
            
            # 构建标准化的数据项
            result_item = {
                "scenario": scenario,
                "id": item["id"],
                "original_instruction": item["original_instruction"],
                "professional_harmful_intent": item["professional_harmful_intent"],
                "category": item["category"],
                "role": item["role"],
                "mind_map_id": mind_map_id,
                "mind_map_theme": map_info.get("mind_map_theme", ""),
                "prompt_template": item["prompt_template"],
                "mindmap_image": mindmap_image_path,
                "keywords": item.get("keywords", "")
            }
            
            result.append(result_item)
    
    return result

def build_ra_attack_prompt(data_item, include_mindmap=True):
   
    if include_mindmap:
           prompt = data_item['prompt_template']
    else:
        prompt = data_item['prompt_template']
    
    return prompt
