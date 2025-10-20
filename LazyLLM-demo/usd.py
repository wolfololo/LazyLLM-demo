import os
import json
import re
import argparse
from typing import List
from PyPDF2 import PdfReader
from docx import Document
from lazyllm import ReactAgent, fc_register, LOG, OnlineChatModule, WebModule

# -------------------------- 1. 核心配置（新增智能提取规则） --------------------------
PAPER_DIR = "DOCS"
SUPPORTED_FORMATS = [".pdf", ".docx", ".doc", ".txt"]
TITLE_PRIORITY_RULE = "标题提取优先级：1. PDF首页顶部/居中文字；2. 摘要 section 上方文字；3. 正文第一部分标题；4. 文件名（仅作为最后 fallback）"

# 伪代码配置（仅YOLO生效）
PSEUDO_CODE_CONFIG = {
    "target_framework": "PyTorch",
    "code_style": "简洁伪代码（保留__init__/forward核心方法）",
    "key_requirement": "严格对应YOLO创新点（backbone/neck/head）"
}

# 非YOLO论文关键词库（用于字段智能提取）
FIELD_EXTRACTION_KEYWORDS = {
    "innovation": ["innovation", "novel", "propose", "proposed", "breakthrough", "improvement"],
    "math": ["formula", "equation", "derivation", "mathematical", "algorithm", "loss function"],
    "dataset": ["dataset", "data set", "corpus", "benchmark", "training data", "test data"],
    "environment": ["python", "pytorch", "tensorflow", "version", "dependency", "library"],
    "hardware": ["gpu", "tpu", "a100", "v100", "rtx", "memory", "ram"],
    "code": ["github", "gitlab", "repository", "open source", "code available", "link:"],
    "comparison": ["compare", "comparison", "baseline", "method", "state-of-the-art", "sota"],
    "metric": ["metric", "accuracy", "perplexity", "bleu", "map", "f1", "precision", "recall"],
    "result": ["result", "performance", "score", "better than", "higher than", "lower than"],
    "conclusion": ["conclusion", "find", "finding", "conclude", "show that", "demonstrate"]
}

CORE_FIELDS = {
    "paper_title": f"论文完整标题（严格按{TITLE_PRIORITY_RULE}提取，禁止编造）",
    "is_yolo_related": "是/否（基于标题+内容关键词）",
    "innovation_point": {
        "core_innovation": "核心创新（如模型/算法/实验设计，150字内）",
        "innovation_value": "创新价值（解决的问题/学术/应用价值，100字内）",
        "pseudo_code": "非YOLO系列论文，无需生成伪代码"
    },
    "math_derivation": {
        "key_formulas": "核心公式（LaTeX格式，如$L = \\frac{1}{N}\\sum (y - \\hat{y})^2$）",
        "derivation_steps": "推导步骤（分点：假设→变换→结论）",
        "math_advantage": "数学优势（对比传统方法，如计算量降低）"
    },
    "reproduction_steps": {
        "data_prep": "数据集（名称+获取地址+预处理）",
        "env_config": "环境（Python版本+核心依赖，如Python3.9+torch==2.2.0）",
        "hardware_req": "硬件（GPU型号/内存，如NVIDIA A100+128GB RAM）",
        "core_steps": "复现流程（分3-5步，如数据加载→训练→验证）",
        "code_info": "代码（开源地址/未开源）"
    },
    "comparison_experiments": {
        "compared_methods": "对比方法（如ResNet-50、BERT-base）",
        "evaluation_metrics": "评价指标（如准确率、BLEU-4）",
        "key_results": "核心结果（如准确率：本文92.5%>ResNet-50 89.3%）",
        "experiment_conclusion": "实验结论（方法优势场景，100字内）"
    }
}


# -------------------------- 2. 论文读取工具（增强内容结构化） --------------------------
def read_pdf(file_path: str) -> dict:
    """优化PDF读取：返回结构化内容（标题候选+摘要+实验+结论），便于后续提取"""
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        content = {
            "total_pages": total_pages,
            "home_page": "",  # 首页内容（标题关键区）
            "abstract": "",  # 摘要（创新点关键区）
            "experiments": "",  # 实验部分（复现/对比关键区）
            "conclusion": "",  # 结论部分（结论关键区）
            "full_text": ""  # 完整文本（ fallback 用）
        }

        # 提取首页（标题）
        if total_pages >= 1:
            home_page_text = reader.pages[0].extract_text() or ""
            content["home_page"] = home_page_text.strip()
            content["full_text"] += f"【首页】\n{home_page_text}\n"

        # 提取前10页（覆盖摘要、实验、结论）
        for page_num in range(min(10, total_pages)):
            page_text = reader.pages[page_num].extract_text() or ""
            page_text_lower = page_text.lower()
            content["full_text"] += f"【第{page_num + 1}页】\n{page_text}\n"

            # 识别摘要（含"abstract"关键词）
            if "abstract" in page_text_lower and not content["abstract"]:
                # 提取"abstract"后的内容（直到下一个section）
                abstract_start = page_text_lower.find("abstract") + len("abstract")
                content["abstract"] = page_text[abstract_start:].strip().split("\n\n")[0]  # 取第一段

            # 识别实验部分（含"experiment"、"result"关键词）
            if any(kw in page_text_lower for kw in ["experiment", "result", "evaluation"]) and len(
                    content["experiments"]) < 5000:
                content["experiments"] += f"【第{page_num + 1}页实验内容】\n{page_text[:2000]}\n"  # 截取关键部分

            # 识别结论部分（含"conclusion"关键词）
            if "conclusion" in page_text_lower and not content["conclusion"]:
                conclusion_start = page_text_lower.find("conclusion") + len("conclusion")
                content["conclusion"] = page_text[conclusion_start:].strip().split("\n\n")[0]

        return content
    except Exception as e:
        return {"error": f"PDF读取失败：{str(e)[:200]}"}


def read_word(file_path: str) -> dict:
    """优化Word读取：结构化提取标题、摘要、实验"""
    try:
        doc = Document(file_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        content = {
            "total_paragraphs": len(paragraphs),
            "home_page": "\n".join(paragraphs[:5]),  # 前5段（标题候选）
            "abstract": "",
            "experiments": "",
            "conclusion": "",
            "full_text": "\n".join(paragraphs[:50])  # 前50段（关键内容）
        }

        # 识别摘要
        for i, para in enumerate(paragraphs):
            if "abstract" in para.lower() and i < len(paragraphs) - 3:
                content["abstract"] = "\n".join(paragraphs[i:i + 4])  # 取后续3段作为摘要
                break

        # 识别实验
        for i, para in enumerate(paragraphs):
            if any(kw in para.lower() for kw in ["experiment", "result"]) and i < len(paragraphs) - 5:
                content["experiments"] += "\n".join(paragraphs[i:i + 6])  # 取后续5段作为实验内容
                if len(content["experiments"]) > 3000:
                    break

        return content
    except Exception as e:
        return {"error": f"Word读取失败：{str(e)[:200]}"}


def read_txt(file_path: str) -> dict:
    """优化TXT读取：结构化提取"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        content = {
            "total_lines": len(lines),
            "home_page": "\n".join(lines[:10]),  # 前10行（标题候选）
            "abstract": "",
            "experiments": "",
            "conclusion": "",
            "full_text": "\n".join(lines[:100])  # 前100行（关键内容）
        }

        # 识别摘要
        for i, line in enumerate(lines):
            if "abstract" in line.lower() and i < len(lines) - 10:
                content["abstract"] = "\n".join(lines[i:i + 11])  # 取后续10行
                break

        # 识别实验
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ["experiment", "result"]) and i < len(lines) - 20:
                content["experiments"] += "\n".join(lines[i:i + 21])
                if len(content["experiments"]) > 3000:
                    break

        return content
    except Exception as e:
        return {"error": f"TXT读取失败：{str(e)[:200]}"}


def get_all_papers() -> List[dict]:
    if not os.path.exists(PAPER_DIR):
        os.makedirs(PAPER_DIR)
        return [{"paper_name": "提示：DOCS文件夹已创建，请将论文放入后重启", "paper_path": "", "status": "empty"}]

    valid_papers = []
    for filename in os.listdir(PAPER_DIR):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in SUPPORTED_FORMATS:
            file_type = "PDF（优先分析）" if file_ext == ".pdf" else "其他格式"
            valid_papers.append({
                "paper_name": f"{filename}（{file_type}）",
                "paper_path": os.path.abspath(os.path.join(PAPER_DIR, filename)),
                "status": "valid",
                "file_ext": file_ext,
                "index": len(valid_papers) + 1
            })

    if not valid_papers:
        return [
            {"paper_name": "DOCS文件夹无支持格式论文（仅支持.pdf/.docx/.doc/.txt）", "paper_path": "", "status": "empty"}]

    return valid_papers


def load_paper_content(paper_path: str, file_ext: str) -> dict:
    """加载结构化内容，区分文件类型"""
    if not os.path.exists(paper_path):
        return {"error": "论文文件不存在（路径错误或已删除）"}

    if file_ext == ".pdf":
        content = read_pdf(paper_path)
    elif file_ext in [".docx", ".doc"]:
        content = read_word(paper_path)
    elif file_ext == ".txt":
        content = read_txt(paper_path)
    else:
        return {"error": f"不支持格式：{file_ext}"}

    # 检查是否读取成功
    if "error" in content:
        return content
    # 确保full_text不为空
    if len(content.get("full_text", "")) < 200:
        return {"error": "论文内容过短（<200字符），无法提取有效信息"}

    return content


# -------------------------- 3. 智能提取工具（核心：无LLM时也能解析字段） --------------------------
def extract_title(structured_content: dict, paper_name: str) -> str:
    """从结构化内容中提取标题（优先级：首页大标题→摘要上方→文件名）"""
    # 1. 从首页提取（优先匹配首行/居中大标题：通常是较长且首字母大写的句子）
    home_page = structured_content.get("home_page", "")
    if home_page:
        # 提取首页前3行，过滤作者/机构信息（含"∗"、"1"、"2"等标记）
        home_lines = [line.strip() for line in home_page.split("\n")[:3] if line.strip()]
        for line in home_lines:
            # 标题特征：长度>5，不含作者标记（∗）、邮箱、机构编号
            if len(line) > 5 and not any(char in line for char in ["∗", "@", "1", "2", "3", "4", "5"]) and \
                    (line.istitle() or line.isupper()):  # 首字母大写或全大写
                return line.strip()

    # 2. 从摘要上方提取
    abstract = structured_content.get("abstract", "")
    if abstract:
        abstract_lines = [line.strip() for line in abstract.split("\n")[:2] if line.strip()]
        for line in abstract_lines:
            if len(line) > 5 and not line.lower().startswith("abstract"):
                return line.strip()

    # 3. 从文件名提取（去除后缀和括号）
    filename_title = re.sub(r'\.(pdf|docx|doc|txt)', '', paper_name.split("（")[0]).strip()
    return f"从文件名提取：{filename_title}"


def extract_innovation(structured_content: dict) -> tuple:
    """从摘要/首页提取创新点（基于关键词匹配）"""
    abstract = structured_content.get("abstract", "")
    home_page = structured_content.get("home_page", "")
    full_text = f"{home_page}\n{abstract}"  # 优先用摘要和首页
    innovation_keywords = FIELD_EXTRACTION_KEYWORDS["innovation"]

    # 提取创新描述（含创新关键词的句子）
    innovation_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in innovation_keywords) and len(sentence) > 10:
            innovation_sentences.append(sentence.strip())

    core_innovation = ""
    innovation_value = ""
    if innovation_sentences:
        # 核心创新：含"propose"、"novel"的句子
        propose_sentences = [s for s in innovation_sentences if "propose" in s.lower() or "novel" in s.lower()]
        core_innovation = propose_sentences[0] if propose_sentences else innovation_sentences[0]
        # 创新价值：含"solve"、"improve"、"enable"的句子
        value_sentences = [s for s in innovation_sentences if
                           any(kw in s.lower() for kw in ["solve", "improve", "enable", "benefit"])]
        innovation_value = value_sentences[0] if value_sentences else "未明确具体价值（需参考全文）"

    # 截断过长内容
    core_innovation = core_innovation[:150] + "..." if len(core_innovation) > 150 else core_innovation
    innovation_value = innovation_value[:100] + "..." if len(innovation_value) > 100 else innovation_value

    return core_innovation or "未明确（需参考全文创新部分）", innovation_value or "未明确（需参考全文价值部分）"


def extract_math(structured_content: dict) -> tuple:
    """从全文提取数学公式和推导（基于关键词和公式特征）"""
    full_text = structured_content.get("full_text", "")
    math_keywords = FIELD_EXTRACTION_KEYWORDS["math"]

    # 1. 提取公式（含"="、"$"、"∑"、"∫"等符号的句子）
    formula_sentences = []
    for line in full_text.split("\n"):
        line_stripped = line.strip()
        if any(char in line_stripped for char in ["=", "$", "∑", "∫", "∂", "∈", "∀", "∃"]) and len(line_stripped) > 3:
            formula_sentences.append(line_stripped)
    key_formulas = "\n".join(formula_sentences[:3]) if formula_sentences else "未明确（需参考全文数学部分）"

    # 2. 提取推导步骤（含"step"、"assume"、"derive"的句子）
    derivation_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in ["step", "assume", "derive", "obtain", "result in"]) and len(
                sentence) > 10:
            derivation_sentences.append(sentence.strip())
    derivation_steps = "\n".join(
        [f"{i + 1}. {s}" for i, s in enumerate(derivation_sentences[:3])]) if derivation_sentences else "未明确（需参考全文推导部分）"

    # 3. 提取数学优势（含"advantage"、"faster"、"lower"的句子）
    advantage_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in ["advantage", "faster", "lower", "reduce", "efficient"]) and len(
                sentence) > 10:
            advantage_sentences.append(sentence.strip())
    math_advantage = advantage_sentences[0] if advantage_sentences else "未明确（需参考全文优势部分）"

    return key_formulas, derivation_steps, math_advantage


def extract_reproduction(structured_content: dict) -> tuple:
    """从实验部分提取复现信息"""
    experiments = structured_content.get("experiments", "")
    full_text = f"{experiments}\n{structured_content.get('full_text', '')}"

    # 1. 数据集提取（含"dataset"关键词）
    dataset_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if "dataset" in sentence.lower() and len(sentence) > 10:
            dataset_sentences.append(sentence.strip())
    data_prep = dataset_sentences[0] if dataset_sentences else "未公开（需参考全文数据集部分）"

    # 2. 环境配置提取（含"python"、"pytorch"关键词）
    env_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in ["python", "pytorch", "tensorflow", "version"]):
            env_sentences.append(sentence.strip())
    env_config = env_sentences[0] if env_sentences else "未公开（需参考全文环境部分）"

    # 3. 硬件提取（含"gpu"、"a100"关键词）
    hardware_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in ["gpu", "tpu", "a100", "v100", "rtx", "memory"]):
            hardware_sentences.append(sentence.strip())
    hardware_req = hardware_sentences[0] if hardware_sentences else "未公开（需参考全文硬件部分）"

    # 4. 复现步骤（基于实验流程逻辑：数据→训练→验证）
    core_steps = [
        "1. 加载数据集并执行预处理（如清洗、归一化）",
        "2. 初始化模型并配置训练参数（如学习率、迭代次数）",
        "3. 执行模型训练并监控训练过程",
        "4. 在测试集上验证模型性能"
    ]
    # 若实验部分有明确步骤，替换默认步骤
    step_sentences = [s.strip() for s in full_text.split("\n") if
                      any(kw in s.lower() for kw in ["step", "train", "test", "load data"])]
    if len(step_sentences) >= 3:
        core_steps = [f"{i + 1}. {s[:80]}..." for i, s in enumerate(step_sentences[:4])]
    core_steps_str = "\n".join(core_steps)

    # 5. 代码提取（含"github"、"open source"关键词）
    code_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in ["github", "open source", "repository", "code"]):
            code_sentences.append(sentence.strip())
    code_info = code_sentences[0] if code_sentences else "未开源（需参考全文代码部分）"

    return data_prep, env_config, hardware_req, core_steps_str, code_info


def extract_comparison(structured_content: dict) -> tuple:
    """从实验部分提取对比实验信息"""
    experiments = structured_content.get("experiments", "")
    conclusion = structured_content.get("conclusion", "")
    full_text = f"{experiments}\n{conclusion}"

    # 1. 对比方法（含"compare"、"baseline"关键词）
    method_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in ["compare", "baseline", "method", "sota"]):
            # 过滤掉不含方法名的句子
            if any(char.isupper() for char in sentence) and len(sentence) > 10:
                method_sentences.append(sentence.strip())
    compared_methods = ", ".join(
        [s.split(",")[0].strip() for s in method_sentences[:3]]) if method_sentences else "未公开（需参考全文对比部分）"

    # 2. 评价指标（含"metric"、"accuracy"关键词）
    metric_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if any(kw in sentence.lower() for kw in FIELD_EXTRACTION_KEYWORDS["metric"]):
            metric_sentences.append(sentence.strip())
    evaluation_metrics = metric_sentences[0] if metric_sentences else "未明确（需参考全文指标部分）"

    # 3. 核心结果（含"%"、"higher"、"lower"关键词）
    result_sentences = []
    for sentence in re.split(r'[.!?]', full_text):
        if "%" in sentence or any(kw in sentence.lower() for kw in ["higher", "lower", "better", "score"]):
            result_sentences.append(sentence.strip())
    key_results = "\n".join(result_sentences[:2]) if result_sentences else "未明确（需参考全文结果部分）"

    # 4. 实验结论（从结论部分提取）
    experiment_conclusion = conclusion[:100] + "..." if len(conclusion) > 100 else conclusion
    if not experiment_conclusion:
        # 从实验部分提取结论
        conclusion_sentences = [s.strip() for s in full_text.split("\n") if
                                any(kw in s.lower() for kw in FIELD_EXTRACTION_KEYWORDS["conclusion"])]
        experiment_conclusion = conclusion_sentences[0][:100] + "..." if conclusion_sentences else "未明确（需参考全文结论部分）"

    return compared_methods, evaluation_metrics, key_results, experiment_conclusion


def generate_smart_fallback(structured_content: dict, paper_name: str, is_yolo: bool) -> dict:
    """智能Fallback：无需LLM，从结构化内容中解析所有字段"""
    fallback = {
        "paper_title": extract_title(structured_content, paper_name),
        "is_yolo_related": "是" if is_yolo else "否",
        "innovation_point": {
            "core_innovation": "",
            "innovation_value": "",
            "pseudo_code": "非YOLO系列论文，无需生成伪代码"
        },
        "math_derivation": {
            "key_formulas": "",
            "derivation_steps": "",
            "math_advantage": ""
        },
        "reproduction_steps": {
            "data_prep": "",
            "env_config": "",
            "hardware_req": "",
            "core_steps": "",
            "code_info": ""
        },
        "comparison_experiments": {
            "compared_methods": "",
            "evaluation_metrics": "",
            "key_results": "",
            "experiment_conclusion": ""
        }
    }

    # 填充创新点
    fallback["innovation_point"]["core_innovation"], fallback["innovation_point"][
        "innovation_value"] = extract_innovation(structured_content)
    # 填充数学推导
    fallback["math_derivation"]["key_formulas"], fallback["math_derivation"]["derivation_steps"], \
    fallback["math_derivation"]["math_advantage"] = extract_math(structured_content)
    # 填充复现步骤
    fallback["reproduction_steps"]["data_prep"], fallback["reproduction_steps"]["env_config"], \
    fallback["reproduction_steps"]["hardware_req"], fallback["reproduction_steps"]["core_steps"], \
    fallback["reproduction_steps"]["code_info"] = extract_reproduction(structured_content)
    # 填充对比实验
    fallback["comparison_experiments"]["compared_methods"], fallback["comparison_experiments"]["evaluation_metrics"], \
    fallback["comparison_experiments"]["key_results"], fallback["comparison_experiments"][
        "experiment_conclusion"] = extract_comparison(structured_content)

    return fallback


def is_yolo_related_paper(structured_content: dict, paper_name: str) -> bool:
    """判断是否为YOLO相关论文"""
    yolo_keywords = ["yolov11", "yolov10", "yolov9", "yolov8", "yolo v11", "yolo v10", "yolo",
                     "object detection", "target detection", "bounding box", "mAP", "FPS"]
    non_yolo_keywords = ["mamba", "diffusion", "time series", "nlp", "language", "transformer", "bert", "gpt", "llm"]

    full_text = structured_content.get("full_text", "").lower()
    paper_name_lower = paper_name.lower()

    # 含非YOLO关键词 → 非YOLO
    if any(kw in full_text for kw in non_yolo_keywords):
        return False
    # 含YOLO关键词 → YOLO
    if any(kw in full_text for kw in yolo_keywords) or any(kw in paper_name_lower for kw in yolo_keywords):
        return True
    # 默认非YOLO
    return False


# -------------------------- 4. LLM分析工具（增强格式约束+智能Fallback） --------------------------
@fc_register("tool")
def batch_analyze_papers() -> str:
    try:
        papers = get_all_papers()
        total = len(papers)
        analyzed = []
        errors = []

        valid_papers = [p for p in papers if p["status"] == "valid"]
        if not valid_papers:
            return json.dumps({
                "overall_status": "error",
                "total_papers": 0,
                "analyzed_papers": [],
                "error_log": [papers[0]["paper_name"]]
            }, ensure_ascii=False, indent=2)

        for idx, paper in enumerate(valid_papers, 1):
            paper_name = paper["paper_name"]
            paper_path = paper["paper_path"]
            file_ext = paper["file_ext"]
            LOG.info(f"正在分析第{idx}/{len(valid_papers)}篇：{paper_name}")

            # 步骤1：加载结构化内容
            structured_content = load_paper_content(paper_path, file_ext)
            if "error" in structured_content:
                errors.append(f"{paper_name}：{structured_content['error']}")
                continue

            # 步骤2：标记论文类型
            is_yolo = is_yolo_related_paper(structured_content, paper_name)
            LOG.info(f"{paper_name}：类型标记为{'YOLO' if is_yolo else '非YOLO'}，开始提取")

            # 步骤3：构造LLM提示词（强制JSON格式+结构化输入）
            analyze_prompt = f"""
            【角色】通用学术论文信息提取专家，需基于结构化内容完整提取所有字段。
            【输入信息】
            - 论文文件名：{paper_name}
            - 结构化内容（优先参考）：
              1. 标题候选：{structured_content.get('home_page', '')[:200]}
              2. 摘要：{structured_content.get('abstract', '')[:500]}
              3. 实验部分：{structured_content.get('experiments', '')[:800]}
              4. 结论：{structured_content.get('conclusion', '')[:300]}
            - 论文类型：{"YOLO系列（需生成伪代码）" if is_yolo else "非YOLO系列（伪代码标固定内容）"}
            - 伪代码规则（仅YOLO生效）：{json.dumps(PSEUDO_CODE_CONFIG, ensure_ascii=False)}

            【输出要求（违反则无效，直接拒绝）】
            1. 仅返回JSON，无任何前置/后置文字（包括"好的"、"以下是"等解释）；
            2. JSON字段必须与以下结构完全一致（字段不能增减，值不能为空白）：
               {{
                   "paper_title": "完整标题（从首页/摘要提取，非文件名）",
                   "is_yolo_related": "是/否",
                   "innovation_point": {{
                       "core_innovation": "核心创新（150字内，含具体改进）",
                       "innovation_value": "创新价值（100字内，含解决的问题）",
                       "pseudo_code": "非YOLO系列论文，无需生成伪代码"  // 非YOLO固定此值
                   }},
                   "math_derivation": {{
                       "key_formulas": "LaTeX格式公式（至少1个，无则标「未明确具体公式」）",
                       "derivation_steps": "分点推导（至少2步，无则标「未明确具体步骤」）",
                       "math_advantage": "数学优势（含对比，无则标「未明确具体优势」）"
                   }},
                   "reproduction_steps": {{
                       "data_prep": "数据集名称+预处理（无则标「未公开具体数据集」）",
                       "env_config": "Python版本+核心依赖（无则标「未公开具体环境」）",
                       "hardware_req": "GPU型号+内存（无则标「未公开具体硬件」）",
                       "core_steps": "分3-5步（无则标「未明确具体步骤」）",
                       "code_info": "GitHub地址或「未开源」"
                   }},
                   "comparison_experiments": {{
                       "compared_methods": "至少1个对比方法（无则标「未公开具体方法」）",
                       "evaluation_metrics": "至少1个指标（无则标「未明确具体指标」）",
                       "key_results": "含数值对比（无则标「未明确具体结果」）",
                       "experiment_conclusion": "100字内结论（无则标「未明确具体结论」）"
                   }}
               }}
            3. 非YOLO论文的"pseudo_code"必须固定为「非YOLO系列论文，无需生成伪代码」，不得修改；
            4. 所有字段值不能为"未明确"，需补充具体描述（如"未公开具体数据集"而非"未公开"）。
            """

            # 步骤4：调用LLM+处理结果（优先LLM，失败则用智能Fallback）
            try:
                llm = OnlineChatModule(
                    source='sensenova',
                    stream=False,
                    mmodel='your_model',
                    api_key='your_api'  # 替换为有效API密钥
                )
                llm_result = llm(analyze_prompt).strip()

                # 清理并解析JSON
                result_json = None
                if llm_result.startswith("```json"):
                    llm_result = llm_result[7:-3].strip()  # 移除代码块标记
                try:
                    result_json = json.loads(llm_result)
                    # 校验非YOLO伪代码
                    if not is_yolo and result_json["innovation_point"][
                        "pseudo_code"] != "非YOLO系列论文，无需生成伪代码":
                        result_json["innovation_point"]["pseudo_code"] = "非YOLO系列论文，无需生成伪代码"
                    # 校验字段非空
                    for field in ["paper_title", "is_yolo_related"]:
                        if not result_json.get(field, "").strip():
                            raise ValueError(f"字段{field}为空")
                except (json.JSONDecodeError, ValueError) as e:
                    LOG.warning(f"{paper_name}：LLM输出无效（{str(e)}），启用智能Fallback")
                    result_json = generate_smart_fallback(structured_content, paper_name, is_yolo)

            except Exception as e:
                LOG.error(f"{paper_name}：LLM调用失败（{str(e)}），启用智能Fallback")
                result_json = generate_smart_fallback(structured_content, paper_name, is_yolo)

            # 添加到分析结果
            analyzed.append({
                "paper_name": paper_name,
                "paper_path": paper_path,
                "is_yolo_related": is_yolo,
                "analysis_result": result_json,
                "extraction_source": "LLM" if "llm_result" in locals() and result_json else "智能提取（无LLM）"
            })

        # 统计结果
        yolo_count = sum(1 for p in analyzed if p["is_yolo_related"])
        non_yolo_count = len(analyzed) - yolo_count
        overall_status = "success" if not errors else "partial_error" if analyzed else "error"

        return json.dumps({
            "overall_status": overall_status,
            "total_papers": total,
            "valid_papers_count": len(valid_papers),
            "analyzed_count": len(analyzed),
            "yolo_related_count": yolo_count,
            "non_yolo_related_count": non_yolo_count,
            "analyzed_papers": analyzed,
            "error_log": errors
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"批量分析失败：{str(e)[:512]}"
        LOG.error(error_msg)
        return json.dumps({
            "overall_status": "error",
            "total_papers": 0,
            "analyzed_papers": [],
            "error_log": [error_msg]
        }, ensure_ascii=False)


# 单篇论文分析（同步增强）
def analyze_single_paper(paper: dict) -> dict:
    paper_name = paper["paper_name"]
    paper_path = paper["paper_path"]
    file_ext = paper["file_ext"]
    print(f"\n=== 开始分析论文：{paper_name} ===")

    # 步骤1：加载结构化内容
    structured_content = load_paper_content(paper_path, file_ext)
    if "error" in structured_content:
        return {"status": "error", "msg": f"内容加载失败：{structured_content['error']}"}

    # 步骤2：标记论文类型
    is_yolo = is_yolo_related_paper(structured_content, paper_name)
    print(f"📌 论文类型：{'YOLO系列（生成伪代码）' if is_yolo else '非YOLO系列（完整提取）'}")

    # 步骤3：调用LLM或智能提取
    try:
        # 构造提示词（同批量分析）
        analyze_prompt = f"""
        【角色】通用学术论文信息提取专家，需完整提取所有字段。
        【输入信息】
        - 论文文件名：{paper_name}
        - 结构化内容：
          标题候选：{structured_content.get('home_page', '')[:200]}
          摘要：{structured_content.get('abstract', '')[:500]}
          实验：{structured_content.get('experiments', '')[:800]}
        - 论文类型：{"YOLO" if is_yolo else "非YOLO"}
        【输出要求】仅返回JSON，字段与CORE_FIELDS完全一致，非YOLO伪代码固定。
        """

        llm = OnlineChatModule(
            source='sensenova',
            stream=False,
            model='your_model',
            api_key='your_api'
        )
        llm_result = llm(analyze_prompt).strip()
        if llm_result.startswith("```json"):
            llm_result = llm_result[7:-3].strip()
        result_json = json.loads(llm_result)
        # 校验非YOLO伪代码
        if not is_yolo:
            result_json["innovation_point"]["pseudo_code"] = "非YOLO系列论文，无需生成伪代码"
        extraction_source = "LLM"

    except Exception as e:
        print(f"⚠️ LLM分析失败（{str(e)[:100]}），启用智能提取")
        result_json = generate_smart_fallback(structured_content, paper_name, is_yolo)
        extraction_source = "智能提取（无LLM）"

    return {
        "status": "success",
        "paper_info": paper,
        "is_yolo_related": is_yolo,
        "extraction_source": extraction_source,
        "analysis_result": result_json
    }


# -------------------------- 5. 本地终端交互（适配智能提取结果展示） --------------------------
def print_terminal_menu():
    print("\n" + "=" * 60)
    print("        通用学术论文分析助手（智能提取版）")
    print("=" * 60)
    print("1. 查看DOCS文件夹中的论文列表")
    print("2. 分析指定单篇论文（LLM+智能Fallback双保障）")
    print("3. 批量分析所有有效论文")
    print("4. 导出单篇论文分析结果（JSON格式）")
    print("5. 退出程序")
    print("=" * 60)


def list_papers_terminal():
    papers = get_all_papers()
    if not papers or papers[0]["status"] == "empty":
        print(f"\n⚠️  {papers[0]['paper_name']}")
        return None

    print("\n📄 DOCS文件夹中的有效论文列表：")
    print("-" * 90)
    print(f"{'索引':<5} {'论文文件名':<60} {'文件类型':<10} {'状态':<10}")
    print("-" * 90)
    for paper in papers:
        if paper["status"] == "valid":
            file_type = "PDF" if paper["file_ext"] == ".pdf" else "其他"
            print(f"{paper['index']:<5} {paper['paper_name'].split('（')[0]:<60} {file_type:<10} 可分析")
    return papers


def export_analysis_result(result: dict, output_dir: str = "ANALYSIS_OUTPUT"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paper_name = result["paper_info"]["paper_name"].split("（")[0].replace(".", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"analysis_{paper_name}_{result['extraction_source']}.json")

    export_data = {
        "export_info": {
            "source": result["extraction_source"],
            "time": "2025年（终端版无实时时间）"
        },
        "paper_info": result["paper_info"],
        "is_yolo_related": result["is_yolo_related"],
        "analysis_result": result["analysis_result"]
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已导出至：{output_path}")
        return output_path
    except Exception as e:
        print(f"\n❌ 导出失败：{str(e)}")
        return None


def terminal_interaction():
    current_analysis_result = None

    while True:
        print_terminal_menu()
        choice = input("\n请输入操作编号（1-5）：").strip()

        if choice == "1":
            list_papers_terminal()

        elif choice == "2":
            papers = list_papers_terminal()
            if not papers or papers[0]["status"] == "empty":
                continue

            try:
                paper_idx = int(input("\n请输入要分析的论文索引：").strip())
                target_paper = next(p for p in papers if p["status"] == "valid" and p["index"] == paper_idx)
            except (ValueError, StopIteration):
                print("\n❌ 无效索引，请重新选择！")
                continue

            # 执行分析
            result = analyze_single_paper(target_paper)
            if result["status"] == "error":
                print(f"\n❌ {result['msg']}")
                continue

            # 保存当前结果（用于导出）
            current_analysis_result = result
            analysis = result["analysis_result"]
            is_yolo = result["is_yolo_related"]
            source = result["extraction_source"]

            # 格式化展示（分模块，清晰易读）
            print(f"\n📊 分析完成（提取来源：{source}）")
            print("=" * 80)
            print(f"1. 论文标题：{analysis['paper_title']}")
            print(f"2. 论文类型：{'✅ YOLO系列' if is_yolo else '🔍 非YOLO系列'}")
            print("\n" + "-" * 80)
            print("3. 核心创新点")
            print("-" * 80)
            print(f"   突破点：{analysis['innovation_point']['core_innovation']}")
            print(f"   价值：{analysis['innovation_point']['innovation_value']}")
            print(f"   伪代码：{analysis['innovation_point']['pseudo_code']}")
            print("\n" + "-" * 80)
            print("4. 数学推导")
            print("-" * 80)
            print(f"   核心公式：\n{analysis['math_derivation']['key_formulas']}")
            print(f"   推导步骤：\n{analysis['math_derivation']['derivation_steps']}")
            print(f"   数学优势：{analysis['math_derivation']['math_advantage']}")
            print("\n" + "-" * 80)
            print("5. 复现步骤")
            print("-" * 80)
            print(f"   数据集：{analysis['reproduction_steps']['data_prep']}")
            print(f"   环境配置：{analysis['reproduction_steps']['env_config']}")
            print(f"   硬件要求：{analysis['reproduction_steps']['hardware_req']}")
            print(f"   核心流程：\n{analysis['reproduction_steps']['core_steps']}")
            print(f"   代码获取：{analysis['reproduction_steps']['code_info']}")
            print("\n" + "-" * 80)
            print("6. 对比实验")
            print("-" * 80)
            print(f"   对比方法：{analysis['comparison_experiments']['compared_methods']}")
            print(f"   评价指标：{analysis['comparison_experiments']['evaluation_metrics']}")
            print(f"   核心结果：\n{analysis['comparison_experiments']['key_results']}")
            print(f"   实验结论：{analysis['comparison_experiments']['experiment_conclusion']}")
            print("=" * 80)

        elif choice == "3":
            print("\n⚠️  批量分析中，请勿关闭程序...（每篇论文约30秒）")
            batch_result = json.loads(batch_analyze_papers())

            print("\n" + "=" * 60)
            print("        批量分析总结报告")
            print("=" * 60)
            print(f"📊 整体统计：")
            print(f"   - 总论文数：{batch_result['total_papers']}")
            print(f"   - 有效论文数：{batch_result['valid_papers_count']}")
            print(f"   - 成功分析数：{batch_result['analyzed_count']}")
            print(f"   - YOLO系列：{batch_result['yolo_related_count']}篇")
            print(f"   - 非YOLO系列：{batch_result['non_yolo_related_count']}篇")
            print(f"   - 分析状态：{batch_result['overall_status']}")

            if batch_result["analyzed_papers"]:
                print(f"\n📄 分析详情（前5篇）：")
                for i, p in enumerate(batch_result["analyzed_papers"][:5], 1):
                    title = p["analysis_result"]["paper_title"][:50] + "..." if len(
                        p["analysis_result"]["paper_title"]) > 50 else p["analysis_result"]["paper_title"]
                    print(f"   {i}. 论文：{p['paper_name'].split('（')[0]}")
                    print(f"      标题：{title}")
                    print(f"      类型：{'YOLO' if p['is_yolo_related'] else '非YOLO'} | 来源：{p['extraction_source']}")

            if batch_result["error_log"]:
                print(f"\n❌ 错误记录（共{len(batch_result['error_log'])}条）：")
                for idx, err in enumerate(batch_result["error_log"], 1):
                    print(f"   {idx}. {err}")
            print("=" * 60)

        elif choice == "4":
            if not current_analysis_result or current_analysis_result["status"] != "success":
                print("\n❌ 无可用分析结果，请先执行「2. 分析指定单篇论文」！")
                continue
            export_analysis_result(current_analysis_result)

        elif choice == "5":
            print("\n👋 感谢使用，程序已退出！")
            break

        else:
            print("\n❌ 无效操作编号，请输入1-5之间的数字！")


# -------------------------- 6. 程序入口（双模式支持） --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用学术论文分析工具（LLM+智能Fallback）")
    parser.add_argument("--mode", type=str, default="terminal", choices=["terminal", "web"],
                        help="运行模式：terminal（本地终端）/ web（Web服务）")
    args = parser.parse_args()

    try:
        if args.mode == "terminal":
            print("🎉 本地终端模式启动成功！支持LLM分析+智能Fallback（无LLM也能提取）")
            terminal_interaction()

        elif args.mode == "web":
            batch_agent = ReactAgent(
                llm=OnlineChatModule(
                    source='sensenova',
                    stream=False,
                    model='your_model',
                    api_key='your_api'
                ),
                tools=['batch_analyze_papers'],
                prompt=f"""
                【角色】通用学术论文分析助手，需展示完整分析结果。
                【展示要求】
                1. 先显示批量总结（总论文数/YOLO数/非YOLO数/错误数）；
                2. 每篇论文按「标题→类型→创新点→数学→复现→对比」顺序展示；
                3. 标注提取来源（LLM/智能提取）；
                4. 非YOLO论文伪代码固定，无需修改。
                """,
                stream=False
            )
            LOG.info("通用学术论文分析Agent初始化完成")

            web_app = WebModule(
                batch_agent,
                port=8847,
                title="通用学术论文分析助手（LLM+智能Fallback）"
            )
            LOG.info("Web服务启动成功，访问地址：http://localhost:8847")
            web_app.start().wait()

    except Exception as e:
        LOG.error(f"程序启动失败：{str(e)}")
        print(
            f"启动错误：{str(e)}\n请检查：\n1. API密钥是否正确；\n2. 依赖库是否安装（pip install PyPDF2 python-docx lazyllm argparse）；\n3. 端口8847是否被占用（Web模式）；\n4. 论文是否为支持格式")