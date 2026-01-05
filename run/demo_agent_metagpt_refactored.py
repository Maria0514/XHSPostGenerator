"""
Streamlit + LangGraph 版本的小红书文案生成器（重构版）
使用模块化的 Node 结构，便于维护和扩展
"""

import os
import json
import re

# 禁用 ChromaDB 遥测功能，避免 telemetry 报错
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import Any, Dict, List
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from tavily import TavilyClient

from spoil import TIANJI_PATH
from spoil.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

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


def build_retrievers(force: bool = False):
    """构建各场景的检索器"""
    embeddings = get_embeddings()
    retrievers: Dict[str, Any] = {}
    dest = os.path.join(TIANJI_PATH, "temp", "tianji-chinese")

    def _sanitize_jsonl_line(s: str) -> str:
        """将常见的非法控制字符转义成 JSON 可解析形式。

        常见来源：把文案直接粘贴进 jsonl，里面夹了真实的 Tab(\t) 等控制字符。
        这些字符在 JSON 字符串里必须写成转义序列（例如 \\t）。
        """
        # 真实制表符会导致 json.loads: Invalid control character
        s = s.replace("\t", "")
        # 其他不可见控制字符（0x00-0x1F，保留合法空白字符 0x09/0x0A/0x0D）统一替换为空格
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
        return s
    
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

        # jsonl-only：每个场景目录下固定放置 examples.jsonl
        jsonl_path = os.path.join(data_path, "examples.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"未找到 jsonl 语料，跳过 {data_path}（需要 {jsonl_path}）")
            continue
        persist = os.path.join(TIANJI_PATH, "temp", f"chromadb_{folder}")
        
        if os.path.exists(persist) and not force:
            vectordb = Chroma(persist_directory=persist, embedding_function=embeddings)
        else:
            if force and os.path.exists(persist):
                import shutil
                shutil.rmtree(persist)

            raw_docs: List[Document] = []
            try:
                with open(jsonl_path, "r", encoding="utf-8-sig") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(_sanitize_jsonl_line(line))
                        except Exception as exc:
                            print(
                                f"jsonl 解析失败，跳过 {jsonl_path}#L{line_no}：{type(exc).__name__}({exc})"
                            )
                            continue

                        text = (obj.get("text") or "").strip()
                        if not text:
                            continue

                        # Chroma metadata 需要扁平的基础类型，避免嵌套 dict/list
                        metadata: Dict[str, Any] = {
                            "scene_id": str(obj.get("scene_id") or scene_id),
                            "id": str(obj.get("id") or f"{folder}-{line_no}"),
                            "type": str(obj.get("type") or "example"),
                            "source": jsonl_path,
                            "line": line_no,
                        }

                        if isinstance(obj.get("scene_name"), str) and obj.get("scene_name").strip():
                            metadata["scene_name"] = obj.get("scene_name").strip()

                        attrs = obj.get("attrs")
                        if isinstance(attrs, dict):
                            for k, v in attrs.items():
                                if isinstance(k, str) and isinstance(v, str) and v.strip():
                                    metadata[f"attr_{k}"] = v.strip()

                        tags = obj.get("tags")
                        if isinstance(tags, list):
                            tag_strs = [str(t).strip() for t in tags if str(t).strip()]
                            if tag_strs:
                                metadata["tags"] = "|".join(tag_strs[:20])

                        raw_docs.append(Document(page_content=text, metadata=metadata))
            except Exception as exc:
                print(f"加载 jsonl 语料失败，跳过 {jsonl_path}: {exc}")
                continue

            # 示例库模式：不做切分，保证召回尽量为“完整单条示例”
            docs = raw_docs
            
            if not docs:
                continue
            
            vectordb = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=persist
            )

        # 使用 MMR 提升检索多样性，减少重复段落
        retrievers[scene_id] = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 5},
        )
    
    return retrievers


# 初始化全局资源
RAG_RETRIEVERS = build_retrievers(force=False)
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
        "chat_completed": False,
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
        if result.get("chat_completed") is True:
            st.session_state["chat_completed"] = True
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_text})
        message(assistant_text, is_user=False, key=f"assistant_{len(st.session_state['chat_history'])}")
        st.rerun()

    if result.get("need_more_info") and not assistant_text:
        fallback = "我需要更多场景要素，请补充信息。"
        st.session_state["chat_history"].append({"role": "assistant", "content": fallback})
        message(fallback, is_user=False, key=f"assistant_{len(st.session_state['chat_history'])}")
        st.rerun()