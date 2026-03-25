# coding: utf-8
import os
import subprocess
import sys
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置常量（自动获取路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(BASE_DIR, "document/txt.txt")  # 自动定位文档
MODEL_PATH = "/home/cy/zzz/Langchain_agent/chatglm_qa/Qwen1.5-1.8B-Chat"  # 直接使用绝对路径
VECTOR_STORE = os.path.join(BASE_DIR, "vector_store")  # 向量库位置

# 全局模型实例
qwen_tokenizer = None
qwen_model = None

def init_qwen_model():
    """初始化Qwen模型"""
    global qwen_tokenizer, qwen_model
    if qwen_model is None:
        print("[INIT] 正在加载Qwen模型...")
        
        try:
            qwen_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
            
            qwen_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto"
            ).eval()
            
            print(f"[INIT] 加载完成 | 设备: {next(qwen_model.parameters()).device}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

def run_retriever(query: str) -> List[str]:
    """改进的检索函数"""
    try:
        enhanced_query = f"{query} 家长 孩子 教育 沟通"
        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "vector_db3.py"),
            "--query", enhanced_query
        ]
        result = subprocess.run(
            ' '.join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return [line.strip() for line in result.stdout.split('\n') if line.strip()]
    except Exception as e:
        print(f"检索失败: {str(e)}")
        return []

def run_qwen(prompt: str) -> str:
    """Qwen生成函数"""
    try:
        init_qwen_model()
        inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
        
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
        return qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"生成失败: {str(e)}"

def build_prompt(question: str, contexts: List[str] = None) -> str:
    """构建提示词"""
    prompt = "你是一个家庭教育专家，请回答以下问题：\n"
    
    if contexts:
        context_items = []
        for i, ctx in enumerate(contexts):
            truncated = (ctx[:300] + '...') if len(ctx) > 300 else ctx
            context_items.append(f"[参考{i+1}] {truncated}")
        
        context_str = "\n".join(context_items)
        prompt += f"\n参考材料：\n{context_str}\n"
    
    prompt += f"\n问题：{question}\n\n请给出：\n1. 核心建议\n2. 具体方法\n3. 注意事项"
    return prompt

def main():
    print("\n" + " 家长沟通助手 ".center(50, "="))
    
    # 初始化向量库
    if not os.path.exists(VECTOR_STORE):
        print("正在初始化向量数据库...")
        run_retriever("初始化测试")
    
    while True:
        try:
            question = input("\n请输入问题（输入'退出'结束）: ").strip()
            if question.lower() in ('退出', 'exit'):
                break
                
            print("\n" + " 检索中... ".center(50, "-"))
            contexts = run_retriever(question)
            
            print("\n" + " 生成中... ".center(50, "-"))
            response = run_qwen(build_prompt(question, contexts))
            
            print("\n" + " 建议 ".center(50, "="))
            print(response.split("问题：")[-1] if response else "生成失败：无有效输出")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n操作已取消")
            break

    print("\n感谢使用！")

if __name__ == "__main__":
    main()