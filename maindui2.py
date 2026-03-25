# coding: utf-8
import os
import subprocess
import sys
import time
from typing import List, Generator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread
from typing import Optional
# 配置常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(BASE_DIR, "document/txt.txt")
MODEL_PATH = "/home/cy/zzz/Langchain_agent/chatglm_qa/Qwen1.5-1.8B-Chat"
VECTOR_STORE = os.path.join(BASE_DIR, "vector_store")

class FamilyEducationAssistant:
    def __init__(self):
        self.qwen_tokenizer = None
        self.qwen_model = None
        self._init_model()

    def _init_model(self):
        """初始化Qwen模型"""
        if self.qwen_model is None:
            print("[INIT] 正在加载Qwen模型...")
            try:
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True
                )
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    device_map="cuda:0",# 测试期间gpu有人占用，故暂时使用cuda:0
                    torch_dtype="auto"
                ).eval()
                print(f"[INIT] 加载完成 | 设备: {next(self.qwen_model.parameters()).device}")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                raise

    def run_retriever(self, query: str) -> List[str]:
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

    def stream_response(self, question: str) -> Generator[str, None, None]:
        """流式生成响应"""
        try:
            # 初始化向量库（如果不存在）
            if not os.path.exists(VECTOR_STORE):
                print("正在初始化向量数据库...")
                self.run_retriever("初始化测试")

            # 检索相关内容
            contexts = self.run_retriever(question)
            
            # 构建提示词
            prompt = self._build_stream_prompt(question, contexts)
            
            # 创建流式处理器
            streamer = TextIteratorStreamer(self.qwen_tokenizer, skip_prompt=True)
            
            # 准备输入
            inputs = self.qwen_tokenizer(
                [prompt], 
                return_tensors="pt",
                padding=True
            ).to(self.qwen_model.device)
            
            # 在单独线程中生成
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True
            )
            thread = Thread(target=self.qwen_model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式输出
            yield "【家庭教育建议】\n"
            for new_text in streamer:
                yield new_text
            
        except Exception as e:
            yield f"[错误] 生成失败: {str(e)}"


    def _build_stream_prompt(self, question: str, contexts: List[str] = None) -> str:
        """构建支持多轮对话的提示词"""
        # 基础提示
        prompt = (
            "你是一位专业的家庭教育顾问，需要根据以下信息和对话历史提供全面、实用的建议。\n\n"
            "### 回答要求：\n"
            "1. 仔细分析问题背景和对话历史，确保回答具有连续性\n"
            "2. 回答分为三个清晰的部分：核心建议、具体方法和注意事项\n"
            "3. 使用简洁明了的语言，避免专业术语\n"
            "4. 参考提供的资料，但避免直接复制\n"
            "5. 考虑不同年龄段孩子的特点\n\n"
        )
        
        # 添加上下文材料
        if contexts:
            prompt += "### 参考资料：\n"
            for i, ctx in enumerate(contexts, 1):
                # 保留关键信息但限制长度
                truncated = ctx[:500] + "..." if len(ctx) > 500 else ctx
                prompt += f"【资料{i}】 {truncated}\n"
            prompt += "\n"
        
        # 添加问题部分，特别强调上下文的重要性
        prompt += (
            f"### 当前问题：\n{question}\n\n"
            "### 回答框架：\n"
            "**核心建议**（1-2条最重要的指导原则）\n"
            "**具体方法**（3-5个可操作步骤，包含示例）\n"
            "**注意事项**（需要避免的常见错误或特殊考虑）\n\n"
            "请根据对话历史和参考资料，提供专业、实用的家庭教育建议："
        )
        
        return prompt