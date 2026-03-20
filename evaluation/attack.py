import sys
sys.modules['flash_attn'] = None
import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
import sys
from pprint import pprint


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_dataset(base_dir, dataset_name, ):
 
    instruction_path = os.path.join(base_dir, 'instruction', f"{dataset_name}.json")
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
        
            # 多模态模式：必须有 ID 且能找到图像
        if IMAGE_ID_KEY not in item or item[IMAGE_ID_KEY] is None:
            print(f"警告: 多模态模式下，数据项 ID {item.get('id')} 缺少 'mind_map_id'，已跳过。")
            continue
                
        image_id = str(item[IMAGE_ID_KEY])
        image_filename = image_id + IMAGE_EXTENSION
        #image_filename = f"{image_id}-4{IMAGE_EXTENSION}"
        full_image_path = os.path.join(mindmap_dir, image_id, image_filename)
            
        if os.path.exists(full_image_path):
            item['mindmap_image'] = full_image_path
            processed_dataset.append(item)
        else:
            print(f"警告: 找不到对应的图片 '{full_image_path}'，已跳过数据项 ID: {item.get('id')}")
        
    
    mode_str = "多模态 (Multi-Modal)"
    print(f"以 {mode_str} 模式从 '{instruction_path}' 成功加载并处理了 {len(processed_dataset)} 条数据。")
    return processed_dataset


def load_model(model_path, model_short_name):
    """
    加载模型的函数，支持 HuggingFace 的模型加载。
    使用类似于成功案例的方式加载模型。
    """
    model_args = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16
    }

    print(f"正在加载模型: {model_short_name} 从路径: {model_path}")
    

    if 'qwen2_5_vl' in model_short_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            #attn_implementation="flash_attention_2",
            **model_args
        ).eval()
    elif 'internvl' in model_short_name.lower() or 'mm-eureka-internvl' in model_short_name.lower():
        model = AutoModel.from_pretrained(
             model_path,
             torch_dtype=torch.bfloat16,
             low_cpu_mem_usage=True,
             #attn_implementation="flash_attention_2",
             trust_remote_code=True,
             device_map="auto",).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    elif 'R1-Onevision' in model_short_name:
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = json.load(f)
                print(config)
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误: {e}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map='auto',
            #attn_implementation="flash_attention_2",
            #attn_implementation="sdpa",
            torch_dtype=torch.bfloat16
        ).eval()
    elif 'llava-cot' in model_short_name.lower():
        model = MllamaForConditionalGeneration.from_pretrained(model_path,
                                                               torch_dtype=torch.bfloat16,
                                                               device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor
    elif 'mm-eureka-qwen' in model_short_name.lower():
        model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True, padding_side="left", use_fast=True)
        return model, processor
    else:
        raise ValueError(f"未知的模型类型: {model_short_name}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    return model, processor


def prepare_inputs(model_short_name, processor, query: str, image_path: str):

    image_obj = Image.open(image_path).convert("RGB")
    print(f"Image size: {image_obj.size}")  
    
    if 'internvl' in model_short_name.lower() or 'mm-eureka-internvl' in model_short_name.lower():
        image = image_path
        if isinstance(image, str):
            image = [image]
        pixel_values_list = []
        for path in image:
            patches = ImageProcessor.load_image(path)
            pixel_values_list.append(patches)
           
        pixel_values = torch.cat(pixel_values_list, dim=0)
        if len(image)>1:
            num_patches_list = [p.size(0) for p in pixel_values_list]
        else:
            num_patches_list = None
     
        if len(image) == 1:
            question = f"<image>\n{query}"
        else:
            placeholders = "\n".join(f"Image-{i+1}: <image>" for i in range(len(image)))
            question = f"{placeholders}\n{query}"

        return {
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,
            'question': question
        }

    elif 'llava-cot' in model_short_name.lower():
            image = image_path
            image = Image.open(image) 
            messages = [
                {'role': 'user', 'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': query}
                ]}
            ]
            prompt  = processor.apply_chat_template(messages, add_generation_prompt=True)
            return prompt, image
    
    elif 'r1-onevision' in model_short_name.lower():
        
        
        MAX_SIZE = 1024
        if max(img.size) > MAX_SIZE:
             ratio = MAX_SIZE / max(img.size)
             new_size = tuple(int(dim * ratio) for dim in img.size)
             img = img.resize(new_size, Image.LANCZOS)
             print(f"图像调整: {Image.open(image_path).size} -> {img.size}")
        
        
             tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
             os.close(tmp_fd)
             img.save(tmp_path)
             image_path = tmp_path
    
        
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        return text, image_inputs
    elif 'mm-eureka-qwen' in model_short_name.lower():
        image = image_path
        placeholders = [{"type": "image", "image": image}]  
        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": query}
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        stop_token_ids = None
        return prompt, image_inputs, stop_token_ids
    else:  
        messages = [{'role': 'user', 'content': f"<image>{query}"}]  
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=[image_obj], padding=True, return_tensors="pt")


def generate(model, model_short_name, processor, query, image, device="cuda", ):
    
    if 'internvl' in model_short_name.lower() or 'mm-eureka-internvl' in model_short_name.lower():
        input_data = prepare_inputs(model_short_name, processor, query, image, )
        
        generation_config = dict(max_new_tokens=4096, do_sample=False)
        
        # --- CORE CORRECTION ---
        chat_args = {
            'tokenizer': processor,
            'question': input_data['question'],
            'pixel_values': None, # Start with None
            'generation_config': generation_config,
            'return_history': False
        }
        
        if input_data['pixel_values'] is not None:
             chat_args['pixel_values'] = input_data['pixel_values'].to(model.device, dtype=model.dtype)
        response = model.chat(**chat_args)
        return response

    elif 'mm-eureka-qwen' in model_short_name.lower():
        text_inputs, image_inputs, stop_token_ids = prepare_inputs(model_short_name, processor, query, image, )
        inputs = processor(
            text=[text_inputs],
            images=image_inputs, 
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=4096,do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    elif 'llava-cot' in model_short_name.lower():
        text_inputs, image_inputs = prepare_inputs(model_short_name, processor, query, image, )
        inputs = processor(text=text_inputs, images=image_inputs, return_tensors="pt").to(model.device) 
        output = model.generate(**inputs, max_new_tokens=4096,do_sample=False,)
        output_text = processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')
        return output_text
    elif 'r1-onevision' in model_short_name.lower():
        text_inputs, image_inputs = prepare_inputs(model_short_name, processor, query, image, )
        inputs = processor(
            text=[text_inputs],
            images=image_inputs, 
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=4096,top_k=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    else:  
        inputs = prepare_inputs(model_short_name, processor, query, image,) # images 会是 None
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response


def process_single_dataset(model, processor, model_path, model_short_name, base_input_dir, dataset_name_json, output_root_dir):
    
   

    output_dir_for_category = os.path.join(output_root_dir, dataset_name_json)
    os.makedirs(output_dir_for_category, exist_ok=True)
    output_file = os.path.join(output_dir_for_category, f"{model_short_name}.jsonl")
    
    
    dataset = load_dataset(base_input_dir, dataset_name_json)
    
    with open(output_file, "w", encoding="utf-8") as fout:
        for item in tqdm(dataset, desc=f"在 {dataset_name_json} 上处理查询"):
            query = item.get("prompt_template")
            
            if not query:
                print(f"警告: 数据项 ID {item.get('id')} 的 'prompt_template' 字段为空，已跳过。")
                continue
            
            
            image_path = item.get("mindmap_image") # 使用 .get() 避免 KeyErrors
            
            
            if not image_path:
                print(f"警告: 多模态模式下，数据项 ID {item.get('id')} 缺少图像路径，已跳过。")
                continue
            
            try:
                
                response = generate(model, model_short_name, processor, query, image_path)
                
                
                result = {
                    "scenario": item.get("scenario"), 
                    "id": item.get("id"), 
                    "original_instruction": item.get("original_instruction"), 
                    "category": item.get("category"), 
                    "mind_map_id": item.get("mind_map_id"), 
                    "query": query, 
                    "image": image_path, 
                    "response": response
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\n错误: 处理数据项 ID {item.get('id')} 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                error_result = {
                    "scenario": item.get("scenario"), 
                    "id": item.get("id"), 
                    "original_instruction": item.get("original_instruction"), 
                    "category": item.get("category"), 
                    "mind_map_id": item.get("mind_map_id"), 
                    "query": query, 
                    "image": image_path, 
                    "response": f"!!GENERATION_ERROR!!: {e}"
                }
                fout.write(json.dumps(error_result, ensure_ascii=False) + "\n")
    print(f"类别 '{dataset_name_json}' 处理完毕。结果已保存到 '{output_file}'。")


def main(args):
    model_path_map = {
        'internvl2_5_8b': '',
        'llava-cot':'',
        'mm-eureka-internvl':'',
        'qwen2_5_vl':'',
        'R1-Onevision':'',
        'mm-eureka-qwen':'',
        'Llama-3.2-11b-V':'',
        'qwen2_vl_7b': ''
    }
    model_short_name = args.model_name
    model_full_path = model_path_map[model_short_name]
    model, processor = load_model(model_full_path, model_short_name)
    
    if args.dataset.lower() == 'all':
        print(f"模式: Z正在处理 '{args.input_dir}' 中的所有数据集...")
        instruction_root = os.path.join(args.input_dir, 'instruction')
        if not os.path.isdir(instruction_root):
            print(f"错误: 在 '{args.input_dir}' 中没有找到 'instruction' 目录。请检查路径。")
            return
        json_files = [f for f in os.listdir(instruction_root) if f.endswith('.json')]
        if not json_files:
            print(f"警告: 在 '{instruction_root}' 中找不到 .json 指令文件。")
            return
        
        dataset_names = [os.path.splitext(f)[0] for f in json_files]
        print(f"发现数据集: {dataset_names}")
        
        for dataset_name in dataset_names:
            print(f"\n{'='*20} 开始处理数据集: {dataset_name} {'='*20}")
            process_single_dataset(model, processor, model_full_path, model_short_name, args.input_dir, dataset_name, args.output_dir)
    else:
        print(f"模式: 正在处理单个数据集 '{args.dataset}'...")
        process_single_dataset(model, processor, model_full_path, model_short_name, args.input_dir, args.dataset, args.output_dir)
        
    print(f"\n脚本执行完毕。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mm-eureka-internvl', help='要使用的模型名称')
    parser.add_argument('--dataset', default="advbench", help="要评估的数据集名称。使用 'all' 来自动处理输入目录下的所有 instruction/*.json 文件。")
    parser.add_argument('--input_dir', default="./Advbench", help="包含 'instruction' 和 'mindmap' 目录的数据集根目录。")
    parser.add_argument('--output_dir', default="./data_RA_response", help='保存输出jsonl文件的根目录')
    args = parser.parse_args()
    print("脚本启动，使用以下参数:")
    pprint(vars(args))
    main(args)
