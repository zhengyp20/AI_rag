# coding: utf-8
import os
import subprocess
import sys
import time
from typing import List, Generator, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

# 配置常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(BASE_DIR, "document/txt.txt")
MODEL_PATH = "/home/cy/zzz/Langchain_agent/chatglm_qa/Qwen1.5-1.8B-Chat"
VECTOR_STORE = os.path.join(BASE_DIR, "vector_store")

class FamilyEducationAssistant:
    def __init__(self):
        self.qwen_tokenizer = None
        self.qwen_model = None
        self.vector_store_initialized = False  # 跟踪向量库初始化状态
        self._init_model()
        self._init_vector_store()  # 初始化向量库

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
                    device_map="cuda:0",
                    torch_dtype="auto"
                ).eval()
                print(f"[INIT] 加载完成 | 设备: {next(self.qwen_model.parameters()).device}")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                raise

    def _init_vector_store(self):
        """初始化向量库（如果不存在）"""
        try:
            if not os.path.exists(VECTOR_STORE):
                print("正在初始化向量数据库...")
                self.run_retriever("初始化测试")
                self.vector_store_initialized = True
            else:
                self.vector_store_initialized = True
                print("向量数据库已存在，跳过初始化")
        except Exception as e:
            print(f"向量库初始化失败: {str(e)}")
            self.vector_store_initialized = False

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

    def get_response(self, question: str) -> str:
        """直接获取完整响应"""
        try:
            # 检查向量库是否初始化成功
            if not self.vector_store_initialized:
                return "[错误] 向量库未成功初始化，请检查配置"

            # 检索相关内容
            contexts = self.run_retriever(question)
            
            # 构建提示词
            prompt = self._build_prompt(question, contexts)
            
            # 准备输入
            inputs = self.qwen_tokenizer(
                [prompt], 
                return_tensors="pt",
                padding=True
            ).to(self.qwen_model.device)
            
            # 生成完整响应
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True
            )
            
            # 解码输出
            response = self.qwen_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            
            return "当前状态的沟通建议:\n" + response
            
        except Exception as e:
            return f"[错误] 生成失败: {str(e)}"

    def _build_prompt(self, question: str, contexts: List[str] = None) -> str:
        """构建提示词"""
        prompt = "你是一个家庭教育专家，请回答以下问题(注意回答的简短有效)：\n"
        if contexts:
            context_items = []
            for i, ctx in enumerate(contexts):
                truncated = (ctx[:300] + '...') if len(ctx) > 300 else ctx
                context_items.append(f"[参考{i+1}] {truncated}")
            context_str = "\n".join(context_items)
            prompt += f"\n参考材料：\n{context_str}\n"
        prompt += f"\n问题：{question}\n\n请给出：\n1. 核心建议\n2. 具体方法\n3. 注意事项"
        return prompt

    def stream_response(self, question: str) -> Generator[str, None, None]:
        """流式生成响应"""
        try:
            # 检查向量库是否初始化成功
            if not self.vector_store_initialized:
                yield "[错误] 向量库未成功初始化，请检查配置"
                return

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
            yield "家庭教育建议:\n"
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
                prompt += f"资料{i}: {truncated}\n"
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

    def generate_welcome_message(self, doctor_info: str) -> str:
        """
        根据医生信息生成个性化欢迎词
        
        参数:
        - doctor_info: 医生信息描述文本
        
        返回:
        - 个性化欢迎消息
        """
        
        if not doctor_info:
            return "您好！我是家庭教育助手，请问有什么关于孩子教育的问题需要帮助吗？"
        try:
            # 检查向量库是否初始化成功
            if not self.vector_store_initialized:
                return "[错误] 向量库未成功初始化，请检查配置"

            # 构建提示词
            prompt = (
                "你是医生的AI助手，需要生成个性化欢迎语。\n\n"
                "### 医生信息：\n"
                f"{doctor_info}\n\n"
                "### 规则：\n"
                "1. 仅提取 **医生姓名** 和 **擅长领域**（如眼科、放疗），忽略职称、医院等信息；\n"
                "2. 格式必须为：`你好！我是[医生姓名]的AI助手，有什么关于[擅长领域]的问题我可以帮你解答？`；\n"
                "3. 字数严格≤30字，语气亲切。\n"
            )
            
            # 准备输入
            inputs = self.qwen_tokenizer(
                [prompt], 
                return_tensors="pt",
                padding=True
            ).to(self.qwen_model.device)
            
            # 生成欢迎消息
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.5,
                do_sample=True
            )
            
            # 解码输出
            welcome_msg = self.qwen_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
        
            return welcome_msg.strip()

        except Exception as e:
            return f"[错误] 生成失败: {str(e)}"

    def stream_response_with_history(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        支持标准message格式的多轮对话流式响应
        参数:
        - messages: 消息列表，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        try:
            if not self.vector_store_initialized:
                yield "[错误] 向量库未初始化"
                return
                
            # 验证消息格式
            if not messages or not isinstance(messages, list):
                yield "[错误] 消息格式无效，应为消息列表"
                return
                
            # 提取当前问题（最后一条用户消息）
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                yield "[错误] 未找到用户消息"
                return
                
            current_question = user_messages[-1].get("content", "")
            
            # 检索相关内容
            contexts = self.run_retriever(current_question)
            
            # 构建包含历史对话的提示词
            prompt = self._build_history_prompt(messages, contexts)
            
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
            yield "家庭教育建议\n"
            for new_text in streamer:
                yield new_text
            
        except Exception as e:
            yield f"[错误] 生成失败: {str(e)}"

    def _build_history_prompt(self, messages: List[Dict[str, str]], contexts: List[str] = None) -> str:
        """
        构建包含多轮对话历史的提示词
        参数:
        - messages: 消息列表，格式为 [{"role": "...", "content": "..."}, ...]
        - contexts: 检索到的相关上下文材料
        """
        # 基础提示
        prompt = (
            "你是一位专业的家庭教育顾问，请根据以下对话历史提供建议：\n\n"
            "### 对话历史：\n"
        )
        
        # 添加历史对话（仅保留最近的10条消息）
        recent_messages = messages[-10:]  # 限制历史长度
        
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                prompt += f"用户：{content}\n"
            elif role == "assistant":
                prompt += f"助手：{content}\n"
        
        # 添加上下文材料
        if contexts:
            prompt += "\n### 参考资料：\n"
            for i, ctx in enumerate(contexts, 1):
                # 保留关键信息但限制长度
                truncated = ctx[:500] + "..." if len(ctx) > 500 else ctx
                prompt += f"资料{i} :{truncated}\n"
        
        prompt += (
            "\n### 回答要求：\n"
            "1. 保持回答与对话历史的连贯性\n"
            "2. 回答包含：核心建议、具体方法、注意事项\n"
            "3. 语言简洁，避免专业术语\n"
            "4. 参考提供的资料，但避免直接复制\n\n"
            "请根据以上信息提供专业建议："
        )
        return prompt