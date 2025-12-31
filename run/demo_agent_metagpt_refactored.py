"""
Streamlit + LangGraph 版本的小红书文案生成器（重构版）
使用模块化的 Node 结构，便于维护和扩展
"""

import os

# 禁用 ChromaDB 遥测功能，避免 telemetry 报错
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import Any, Dict
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from tavily import TavilyClient

from spoil import TIANJI_PATH
from spoil.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 导入重构后的模块
from spoil.spoil_agent import build_xhs_workflow
from spoil.spoil_agent.config import RAG_SCENE_MAP, SCENE_OPTIONS
from spoil.spoil_agent.spoilState import SpoilState

load_dotenv()


# --------------- 初始化资源 ---------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """获取 Embedding 模型"""
    return SiliconFlowEmbeddings()


def build_retrievers(chunk_size: int = 896, force: bool = False):
    """构建各场景的检索器"""
    embeddings = get_embeddings()
    retrievers: Dict[str, Any] = {}
    dest = os.path.join(TIANJI_PATH, "temp", "tianji-chinese")
    
    if not os.path.exists(dest):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="sanbu/tianji-chinese",
            local_dir=dest,
            repo_type="dataset",
            local_dir_use_symlinks=False,
            endpoint=os.environ.get("HF_ENDPOINT", None),
        )

    for scene_id, (_, folder) in RAG_SCENE_MAP.items():
        if folder is None:
            continue
        data_path = os.path.join(dest, "RAG", folder)
        if not os.path.exists(data_path):
            continue
        persist = os.path.join(TIANJI_PATH, "temp", f"chromadb_{folder}")
        
        if os.path.exists(persist) and not force:
            vectordb = Chroma(persist_directory=persist, embedding_function=embeddings)
        else:
            if force and os.path.exists(persist):
                import shutil
                shutil.rmtree(persist)
            
            loader = DirectoryLoader(
                data_path,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=200
            )
            try:
                docs = splitter.split_documents(loader.load())
            except Exception as exc:
                print(f"加载知识库失败，跳过 {data_path}: {exc}")
                continue
            
            if not docs:
                continue
            
            vectordb = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=persist
            )
        
        retrievers[scene_id] = vectordb.as_retriever()
    
    return retrievers


# 初始化全局资源
RAG_RETRIEVERS = build_retrievers()
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_CLIENT = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None

@st.cache_resource(show_spinner=False)
def get_app():
    """构建并缓存工作流"""
    return build_xhs_workflow(RAG_RETRIEVERS, TAVILY_CLIENT)

APP = get_app()


# --------------- Streamlit UI ---------------
st.set_page_config(page_title="小红书文案生成器", page_icon="💅🏼")

# 初始化 session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "scene_label" not in st.session_state:
    st.session_state["scene_label"] = ""
if "scene_attributes" not in st.session_state:
    st.session_state["scene_attributes"] = {}
if "enable_se" not in st.session_state:
    st.session_state["enable_se"] = False
if "chat_completed" not in st.session_state:
    st.session_state["chat_completed"] = False


def reset_chat():
    """重置对话"""
    st.session_state["chat_history"] = []
    st.session_state["scene_label"] = ""
    st.session_state["scene_attributes"] = {}


# 侧边栏
with st.sidebar:
    st.markdown("## 📝 支持的内容类型")
    for item in SCENE_OPTIONS:
        st.write(item)
    st.markdown("---")
    st.markdown("### 🎯 当前内容类型")
    st.write(st.session_state["scene_label"])
    st.markdown("### 🔍 文案属性")
    st.write(st.session_state["scene_attributes"])
    st.markdown("---")
    st.checkbox("🌐 启用网络搜索（需要 TAVILY_API_KEY）", key="enable_se")
    st.button("🔄 清空对话", on_click=reset_chat)

# 主界面
st.title("✨ 小红书智能文案生成器")

# 显示聊天历史
for idx, turn in enumerate(st.session_state["chat_history"]):
    if turn["role"] == "user":
        message(turn["content"], is_user=True, key=f"user_{idx}")
    else:
        message(turn["content"], is_user=False, key=f"assistant_{idx}")

# 用户输入处理
if user_input := st.chat_input("💡 告诉我你想创作什么样的文案..."):
    # 如果上一个对话已完成，清空所有数据开启新对话
    if st.session_state.get("chat_completed", False):
        st.session_state["chat_history"] = []
        st.session_state["scene_label"] = ""
        st.session_state["scene_attributes"] = {}
        st.session_state["chat_completed"] = False

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    
    init_state: SpoilState = {
        "user_input": user_input,
        "chat_history": st.session_state["chat_history"],
        "scene_label": st.session_state.get("scene_label", ""),
        "scene_attributes": st.session_state.get("scene_attributes", {}),
        "retrieved_docs": [],
        "search_enabled": st.session_state.get("enable_se", False),
        "search_queries": [],
        "search_results": {},
        "search_context": "",
        "final_answer": "",
        "need_more_info": False,
    }
    
    message(user_input, is_user=True, key=f"user_{len(st.session_state['chat_history'])}")

    with st.spinner("思考中..."):
        result = APP.invoke(init_state)

    if result.get("scene_label"):
        st.session_state["scene_label"] = result["scene_label"]
    if result.get("scene_attributes"):
        st.session_state["scene_attributes"] = result["scene_attributes"]

    assistant_text = result.get("final_answer", "")
    if assistant_text:
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_text})
        message(assistant_text, is_user=False, key=f"assistant_{len(st.session_state['chat_history'])}")
        st.rerun()

    if result.get("need_more_info") and not assistant_text:
        fallback = "我需要更多场景要素，请补充信息。"
        st.session_state["chat_history"].append({"role": "assistant", "content": fallback})
        message(fallback, is_user=False, key=f"assistant_{len(st.session_state['chat_history'])}")
        st.rerun()