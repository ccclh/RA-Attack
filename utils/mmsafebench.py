import os
import json

def read_mm_safebench(dataset_path, img_type="SD_TYPO",query_type = "Rephrased Question(SD)"):
    """
    读取 mm-safebench 数据集，返回每个问题的 query 文本及对应的图片路径。

    参数:
        dataset_path (str): 数据集根目录，例如 "/data/simabingrui/dataset/mm-safebench"
        img_type (str): 图片子文件夹名称，默认使用 "SD"，也可以指定 "SD_TYPO" 或 "TYPO"

    返回:
        List[dict]: 每个字典包含以下信息：
            - scenario: 场景名称（例如 "01-Illegal_Activitiy"）
            - question_id: 问题编号（字符串，如 "0", "1", ...）
            - query: 问题文本，优先选择 "Rephrased Question(SD)"（若 img_type 为 "SD" 且该字段存在），否则依次尝试 "Rephrased Question"、"Changed Question"、"Question"
            - image: 对应图片的完整路径，格式为
                     {dataset_path}/img/{scenario}/{img_type}/{question_id}.jpg
    """
    processed_questions_dir = os.path.join(dataset_path, "processed_questions")
    img_dir = os.path.join(dataset_path, "img")
    data_list = []

    # 允许的场景列表
    allowed_scenarios = {
        "01-Illegal_Activitiy",
        "02-HateSpeech",
        "04-Physical_Harm",
        "06-Fraud",
        "07-Sex",
        "09-Privacy_Violence",
        "10-Legal_Opinion"
    }

    # 遍历 processed_questions 文件夹下所有 JSON 文件
    for json_file in os.listdir(processed_questions_dir):
        if not json_file.endswith(".json"):
            continue
        # 得到场景名称，例如 "01-Illegal_Activitiy"
        scenario = os.path.splitext(json_file)[0]
        
        #if scenario not in allowed_scenarios:
         #   continue  # 跳过不在白名单中的场景

        json_path = os.path.join(processed_questions_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        # 遍历当前 JSON 文件中的每个问题
        for qid, content in questions.items():
            # 选择 query 文本
            query_text = content[query_type]
            harm_insturction = content["Changed Question"]
            # 构造图片的完整路径
            image_path = os.path.join(img_dir, scenario, img_type, f"{qid}.jpg")
            
            data_list.append({
                "scenario": scenario,
                "question_id": qid,
                "ori_harm_query":harm_insturction,
                "query": query_text,
                "image": image_path
            })

    return data_list
