import os
from dashscope import MultiModalConversation


# 配置API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("未找到DASHSCOPE_API_KEY环境变量，请设置后再使用")

# 推荐模型配置（性价比高）
MODEL_CONFIG = {
    # 主推荐：qwen-vl-plus - 性价比高，支持图像理解
    "qwen-vl-plus": {
        "model_name": "qwen-vl-plus",
        "description": "通义千问多模态模型，支持图像理解，性价比高",
        "price_per_1k_tokens": 0.008,  # 输入价格（元/千tokens）
        "capabilities": ["图像理解", "文本生成", "OCR"],
        "max_tokens": 12000,
    },
    
    # 备选1：qwen-vl-max - 更强但稍贵
    "qwen-vl-max": {
        "model_name": "qwen-vl-max",
        "description": "通义千问多模态模型（增强版），性能更强",
        "price_per_1k_tokens": 0.02,  # 输入价格（元/千tokens）
        "capabilities": ["图像理解", "文本生成", "OCR", "复杂推理"],
        "max_tokens": 12000,
    },
    
    # 备选2：qwen2-vl-72b-instruct - 开源模型
    "qwen2-vl-72b-instruct": {
        "model_name": "qwen2-vl-72b-instruct",
        "description": "Qwen2-VL 72B指令微调版，能力强",
        "price_per_1k_tokens": 0.02,
        "capabilities": ["图像理解", "文本生成", "视频理解"],
        "max_tokens": 32768,
    }
}

# 默认使用模型
DEFAULT_MODEL = "qwen-vl-plus"


def get_client():
    """获取DashScope客户端"""
    import dashscope
    dashscope.api_key = DASHSCOPE_API_KEY
    return dashscope


def call_multimodal_model(prompt, image_url=None, model=None):
    """
    调用多模态模型
    
    参数:
        prompt: 文本提示
        image_url: 图片URL或本地路径（可选）
        model: 模型名称，默认使用qwen-vl-plus
    
    返回:
        包含模型响应和调用计价的字典:
        {
            "response": 模型原始响应,
            "content": 模型输出内容,
            "usage": {
                "input_tokens": 输入token数,
                "output_tokens": 输出token数,
                "total_tokens": 总token数
            },
            "cost": {
                "input_cost": 输入费用(元),
                "output_cost": 输出费用(元),
                "total_cost": 总费用(元)
            }
        }
    """
    if model is None:
        model = DEFAULT_MODEL
    
    messages = []
    
    # 构建消息内容
    content = []
    
    # 如果有图片，添加图片内容
    if image_url:
        if image_url.startswith("http"):
            content.append({"image": image_url})
        else:
            content.append({"image": f"file://{image_url}"})
    
    # 添加文本内容
    content.append({"text": prompt})
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    # 调用API
    response = MultiModalConversation.call(
        model=model,
        messages=messages
    )
    
    # 计算调用费用
    cost_info = calculate_cost(response, model)
    
    # 构建返回结果
    result = {
        "response": response,
        "content": extract_content(response),
        "usage": extract_usage(response),
        "cost": cost_info
    }
    
    return result


def extract_content(response):
    """从响应中提取内容"""
    try:
        if hasattr(response, 'output') and response.output:
            if hasattr(response.output, 'choices') and response.output.choices:
                return response.output.choices[0].message.content
        return None
    except:
        return None


def extract_usage(response):
    """从响应中提取usage信息"""
    try:
        if hasattr(response, 'usage') and response.usage:
            return {
                "input_tokens": getattr(response.usage, 'input_tokens', 0),
                "output_tokens": getattr(response.usage, 'output_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0),
            }
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    except:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def calculate_cost(response, model):
    """
    计算调用费用
    
    参数:
        response: API响应
        model: 模型名称
    
    返回:
        费用信息字典
    """
    # 获取模型价格配置
    if model not in MODEL_CONFIG:
        return {"error": f"未找到模型 {model} 的价格配置"}
    
    price_config = MODEL_CONFIG[model]
    price_per_1k = price_config["price_per_1k_tokens"]
    
    # 提取token使用量
    usage = extract_usage(response)
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = usage["total_tokens"]
    
    # 计算费用（元）
    input_cost = (input_tokens / 1000) * price_per_1k
    output_cost = (output_tokens / 1000) * price_per_1k
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "price_per_1k_tokens": price_per_1k,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": "CNY"
    }


# 示例用法
if __name__ == "__main__":
    # 测试配置是否正确
    print("API Key已配置:", DASHSCOPE_API_KEY[:10] + "..." if DASHSCOPE_API_KEY else "未配置")
    print("默认模型:", DEFAULT_MODEL)
    print("模型信息:", MODEL_CONFIG[DEFAULT_MODEL]["description"])
    print("价格:", MODEL_CONFIG[DEFAULT_MODEL]["price_per_1k_tokens"], "元/千tokens")
