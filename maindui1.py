# main.py
import os
import argparse
from typing import List
from core.models import QwenModel
from core.retriever import run_retriever  # 假设检索功能已移到retriever.py

# 初始化模型
qwen = QwenModel(MODEL_PATH)

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

def process_from_file(input_file: str, output_file: str = None):
    """从文件读取问题并处理"""
    print("\n" + " 家长沟通助手 ".center(50, "="))
    
    # 初始化向量库
    if not os.path.exists(VECTOR_STORE):
        print("正在初始化向量数据库...")
        run_retriever("初始化测试")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取输入文件失败: {str(e)}")
        return
    
    # 处理每个问题
    for question in questions:
        try:
            print(f"\n处理问题: {question}")
            
            print("\n" + " 检索中... ".center(50, "-"))
            contexts = run_retriever(question)
            
            print("\n" + " 生成中... ".center(50, "-"))
            prompt = build_prompt(question, contexts)
            
            print("\n" + " 建议 ".center(50, "="))
            # 流式输出
            full_response = ""
            for chunk in stream_qwen_response(prompt):
                print(chunk, end='', flush=True)
                full_response += chunk
                time.sleep(0.05)  # 控制输出速度
            
            print("\n" + "=" * 50)
            
            # 如果需要保存到文件
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"问题: {question}\n")
                    f.write(f"回答: {full_response}\n\n")
                    
        except KeyboardInterrupt:
            print("\n操作已取消")
            break

    print("\n处理完成！")

def process_single_question(question: str) -> str:
    """处理单个问题"""
    try:
        contexts = run_retriever(question)
        prompt = build_prompt(question, contexts)
        return "".join([chunk for chunk in qwen.stream_response(prompt)])
    except Exception as e:
        return f"处理失败: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='家庭教育问答系统')
    parser.add_argument('--input', help='输入问题文件路径')
    parser.add_argument('--output', help='输出结果文件路径')
    parser.add_argument('--api', action='store_true', help='启动API服务模式')
    
    args = parser.parse_args()
    
    if args.api:
        from api_server import start_api
        start_api(qwen)  # 传入模型实例
    elif args.input:
        process_from_file(args.input, args.output)
    else:
        # 交互式模式
        while True:
            question = input("请输入问题(输入q退出): ")
            if question.lower() == 'q':
                break
            print(process_single_question(question))