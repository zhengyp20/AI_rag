# 文件名: combined_family_education_api.py
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Any, Generator
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入家庭教辅系统类
from main_jy import FamilyEducationAssistant

# --- 1. 基础设置 (日志, 线程池) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

process_pool = ThreadPoolExecutor(max_workers=8)

# --- 2. 应用程序生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 家庭教育助手API服务启动中...")
    try:
        # 初始化家庭教辅系统
        app.state.system = FamilyEducationAssistant()
        logger.info("✅ 家庭教育助手系统初始化成功")
        yield  # 应用运行期间保持状态
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 家庭教育助手API服务关闭")
        process_pool.shutdown(wait=False)

# --- 3. FastAPI 应用实例与CORS ---
app = FastAPI(
    title="家庭教育助手API (多轮对话+学生建议查询)",
    description="提供家庭教育相关的多轮对话服务和学生专属查询服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置跨域资源共享
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Pydantic 数据模型 ---
class DialogQueryRequest(BaseModel):
    """多轮对话查询请求"""
    question: str  # 用户问题
    session_id: Optional[str] = None  # 会话ID（可选）
    user_id: Optional[str] = None  # 用户ID（可选）
    # images: Optional[List[str]] = None  # 图片列表（可选）

class StudentQueryRequest(BaseModel):
    """学生查询请求"""
    student_id: str  # 学生ID (必需)
    question: str    # 用户当前状态以及问题
    context: Optional[str] = None  # 可选额外上下文

class SessionResponse(BaseModel):
    """会话响应结构"""
    session_id: str  # 会话ID
    response: str  # 助手的第一条响应

class StudentResponse(BaseModel):
    """学生查询响应结构"""
    student_id: str  # 学生ID
    response: str    # 完整响应内容
    request_id: str  # 请求唯一标识

# --- 5. 会话存储管理 ---
class SessionData:
    """会话数据类，存储对话历史"""
    def __init__(self, session_id: str, user_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.history: List[Dict[str, str]] = []  # 对话历史
        self.created_at = asyncio.get_event_loop().time()
        self.last_accessed = self.created_at
        
    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        # 保留最近的10轮对话
        if len(self.history) >= 20:
            self.history = self.history[-18:]
        self.history.append({"role": role, "content": content})
        self.last_accessed = asyncio.get_event_loop().time()
        
    def get_context(self, max_length: int = 3000) -> str:
        """生成对话上下文，限制最大长度"""
        context = ""
        for msg in self.history[-10:]:  # 只取最近10条消息
            prefix = "用户：" if msg["role"] == "user" else "助手："
            context += f"{prefix}{msg['content']}\n"
            if len(context) > max_length:
                break
        return context

# 会话存储字典 {session_id: SessionData}
sessions: Dict[str, SessionData] = {}
student_requests: Dict[str, dict] = {}  # 学生请求记录

async def cleanup_sessions():
    """定期清理过期会话和请求"""
    while True:
        await asyncio.sleep(300)  # 每5分钟清理一次
        current_time = asyncio.get_event_loop().time()
        expired_sessions = []
        expired_requests = []
        
        # 清理过期会话
        for session_id, session_data in sessions.items():
            if current_time - session_data.last_accessed > 1800:  # 30分钟
                expired_sessions.append(session_id)
        
        # 清理旧请求记录
        max_requests = 500
        if len(student_requests) > max_requests:
            # 删除最早的10%记录
            to_remove = sorted(student_requests.keys())[:int(max_requests * 0.1)]
            expired_requests = to_remove
                
        # 执行清理
        for session_id in expired_sessions:
            logger.info(f"清理过期会话: {session_id}")
            del sessions[session_id]
            
        for req_id in expired_requests:
            logger.info(f"清理过期请求: {req_id}")
            del student_requests[req_id]
                
        logger.info(f"清理完成: {len(expired_sessions)}会话, {len(expired_requests)}请求")

# --- 6. API 端点 ---

# ===== 多轮对话端点 =====
@app.post("/v1/dialog/start_session", response_model=SessionResponse)
async def start_dialog_session(user_id: Optional[str] = None):
    """创建一个新的对话会话"""
    try:
        # 创建新会话
        session_id = f"session_{uuid.uuid4().hex}"
        session_data = SessionData(session_id, user_id)
        sessions[session_id] = session_data
        
        # 生成欢迎消息
        welcome_message = "您好！我是家庭教育助手，请问有什么关于孩子教育的问题需要帮助吗？"
        
        session_data.add_message("assistant", welcome_message)
        
        logger.info(f"创建新会话 [session={session_id}]")
        
        return SessionResponse(
            session_id=session_id,
            response=welcome_message
        )
    except Exception as e:
        logger.error(f"创建会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="创建会话时发生错误")

@app.post("/v1/dialog/query", 
            response_class=StreamingResponse,
            responses={
                    200: {
                    "content": {"text/plain": {}},
                    "description": "流式文本响应"
                   }
    })
async def dialog_query(request: DialogQueryRequest):
    """处理用户查询并流式返回响应，支持多轮对话"""
    try:
        # 会话管理
        session_data = None
        if request.session_id and request.session_id in sessions:
            session_data = sessions[request.session_id]
            logger.info(f"继续现有会话 [session={request.session_id}]")
        else:
            # 创建新会话
            session_id = f"session_{uuid.uuid4().hex}"
            session_data = SessionData(session_id, request.user_id)
            sessions[session_id] = session_data
            logger.info(f"创建新会话 [session={session_id}]")
        
        # 添加用户消息到历史
        user_message = request.question
        # if request.images:
        #     user_message += f" (包含{len(request.images)}张图片)"
        # session_data.add_message("user", user_message)
        
        # 获取对话上下文
        context = session_data.get_context()
        full_question = f"{context}\n问题: {request.question}"
        
        logger.info(f"处理用户查询 [session={session_data.session_id}]: {request.question[:50]}...")
        
        async def response_generator():
            """异步生成器，用于流式返回结果"""
            loop = asyncio.get_event_loop()
            try:
                # 在线程池中运行查询处理
                gen = await loop.run_in_executor(
                    process_pool,
                    app.state.system.stream_response,  # 使用流式响应方法
                    full_question
                )
                
                # 收集完整响应
                full_response = ""
                
                # 流式返回结果
                for chunk in gen:
                    full_response += chunk
                    yield chunk
                
                # 添加助手响应到历史
                session_data.add_message("assistant", full_response)
                
                logger.info(f"会话完成 [session={session_data.session_id}]")
                
            except Exception as e:
                logger.error(f"查询处理失败: {e}", exc_info=True)
                yield "抱歉，处理您的查询时发生错误。"
        
        return StreamingResponse(
            response_generator(),
            media_type="text/plain; charset=utf-8",
            headers={"X-Session-ID": session_data.session_id}
        )
    
    except Exception as e:
        logger.error(f"请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="处理查询时发生错误")

@app.get("/v1/dialog/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取指定会话的历史记录"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session_data = sessions[session_id]
    return {
        "session_id": session_data.session_id,
        "user_id": session_data.user_id,
        "history": session_data.history,
        "created_at": session_data.created_at,
        "last_accessed": session_data.last_accessed
    }

# ===== 学生查询端点 =====
@app.post("/v1/student/query", response_model=StudentResponse)
async def student_query(request: StudentQueryRequest):
    """处理学生查询并返回完整响应"""
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    try:
        logger.info(f"处理学生查询 [student={request.student_id}, request={request_id}]: {request.question[:50]}...")
        
        # 构建完整问题（包含学生ID）
        full_question = f"学生ID: {request.student_id}\n问题: {request.question}"
        if request.context:
            full_question += f"\n额外上下文: {request.context}"
        
        # 在线程池中运行查询处理
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            process_pool,
            app.state.system.get_response,  # 使用非流式响应方法
            full_question
        )
        
        # 记录请求
        student_requests[request_id] = {
            "student_id": request.student_id,
            "question": request.question,
            "context": request.context,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"学生查询完成 [student={request.student_id}, request={request_id}]")
        
        return StudentResponse(
            student_id=request.student_id,
            response=response,
            request_id=request_id
        )
    
    except Exception as e:
        logger.error(f"学生查询处理失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"处理学生查询时发生错误: {str(e)}"
        )

# ===== 辅助端点 =====
@app.get("/v1/student/request/{request_id}")
async def get_request_record(request_id: str):
    """获取学生请求记录"""
    if request_id not in student_requests:
        raise HTTPException(status_code=404, detail="请求记录不存在")
    
    return {
        "request_id": request_id,
        "data": student_requests[request_id]
    }

# --- 7. 健康检查与服务器启动 ---
@app.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "version": app.version,
        "sessions": len(sessions),
        "student_requests": len(student_requests)
    }

@app.get("/")
async def root():
    """根端点，提供基本信息"""
    return {
        "message": "家庭教育助手API服务运行中",
        "version": app.version,
        "endpoints": {
            "dialog": {
                "start_session": "/v1/dialog/start_session (POST)",
                "query": "/v1/dialog/query (POST)",
                "session_history": "/v1/dialog/session/{session_id}/history (GET)"
            },
            "student": {
                "query": "/v1/student/query (POST)",
                "request_record": "/v1/student/request/{request_id} (GET)"
            }
        }
    }

if __name__ == "__main__":
    # 启动会话清理任务
    loop = asyncio.get_event_loop()
    loop.create_task(cleanup_sessions())
    
    PORT = 5500
    logger.info(f"🌐 启动服务器于 http://localhost:{PORT}")
    logger.info(f"📚 API 文档位于 http://localhost:{PORT}/docs")
    
    uvicorn.run(
        "combined_family_education_api:app",
        host="0.0.0.0", 
        port=PORT,
        workers=2,
        reload=True
    )