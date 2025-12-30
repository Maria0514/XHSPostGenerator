"""
Streamlit + LangGraph 版本的天机智能体
全部改为 LangChain/LangGraph，实现 RAG（参考 demo_rag_langchain_all.py）与可选联网搜索。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from tianji.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM, ZhipuLLM

import loguru
from tianji import TIANJI_PATH
from tianji.agents.metagpt_agents.utils.helper_func import (
    extract_all_types,
    extract_all_types_and_examples,
    extract_attribute_descriptions,
    extract_single_type_attributes_and_examples,
    has_empty_values,
    is_number_in_types,
    load_json,
)
from tianji.knowledges.langchain_onlinellm.models import SiliconFlowLLM
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()

logger = loguru.logger


class TianjiState(TypedDict):
    user_input: str
    chat_history: List[Dict[str, str]]
    scene_label: str
    scene_attributes: Dict[str, str]
    retrieved_docs: List[str]
    search_enabled: bool
    search_results: Dict[str, Any]
    final_answer: str
    need_more_info: bool


# --------------- 全局资源 ---------------
SCENE_JSON = load_json("scene_attribute.json")
SCENE_OPTIONS = extract_all_types(SCENE_JSON)
SCENE_EXAMPLES = extract_all_types_and_examples(SCENE_JSON)

INTENT_PROMPT_TEMPLATE = """
#Role: 小红书内容分类助手

## 任务：
根据用户输入，识别用户想要创作的小红书内容类型。

## 内容类型：
1. 生活分享 - 日常趣事、美食、旅游见闻
2. 美妆护肤 - 产品测评、护肤经验
3. 时尚穿搭 - 搭配灵感、风格分享
4. 运动健康 - 健身训练、减肥方法
5. 科技数码 - 电子产品、数码摄影
6. 音乐影视 - 音乐推荐、电视剧推荐
7. 书籍阅读 - 书籍推荐、阅读方法
8. 宠物生活 - 宠物养护、宠物知识

## 约束：
- 只返回数字（1-8），不需要其他内容
- 如果不符合任何类型，返回"None"

## 输入：
用户输入：```{instruction}```
"""

REFINE_PROMPT_TEMPLATE = """
#Role: 小红书文案属性提取助手

## Background:
- 作为一个专业的{scene}内容创作助手，你需要从用户的需求描述中提取创作文案所需的关键属性。

## Goals:
- 从用户的历史对话中分析并提取小红书文案创作所需的所有关键要素，形成结构化的属性信息。

## Constraints:
- 只返回单个 json 对象，不要返回其他内容。
- 如果没有提取到对应的属性请用空字符串表示，例如："目标受众": ""。
- 如果发现属性发生更新，用新值覆盖旧值。
- 属性值应该简洁明了，用关键词表示。

## Input:
- 用户的创作需求：```{instruction}```
- 需要提取的属性: ```{scene_attributes}```
- 每个属性的详细说明:```{scene_attributes_description}```
"""

QUESTION_PROMPT_TEMPLATE = """
#Role: 小红书文案创作助手

## Goals:
- 根据当前已收集的信息，针对缺失的关键属性提出一个自然、友好的追问。

## Constraints:
- 如果所有属性都已完整，回复字符串"Full"。
- 只提一个问题，使用自然对话的语气。
- 问题应该引导用户提供具体、有用的信息。

## Input:
- 文案类型：```{scene}```
- 当前已有的属性: ```{scene_attributes}```
- 各属性的详细说明:```{scene_attributes_description}```
"""

ANSWER_PROMPT_TEMPLATE = """
#Role: {scene}内容创作专家

## 任务：
基于用户需求、创作属性和参考内容，为用户创作一篇优质的小红书文案。

## 创作指南：
1. 开头策略：用表情符号、问题、对比或数据吸引注意力
2. 内容组织：结合故事感和实用干货，逻辑清晰
3. 结尾互动：提出问题、发起投票或话题讨论，引发评论
4. 文案风格：
   - 语气要符合目标受众的审美
   - 适当使用emoji和话题标签（#话题）
   - 避免生硬广告，要有真实感和亲近感
5. 字数控制：300-800字之间

## Constraints:
- 严格遵守用户指定的文案风格和目标受众
- 如果有搜索结果，优先参考最新热点信息
- 结合RAG参考文案的优秀表达方式
- 内容要原创，避免直接复制参考文案
- 确保文案符合小红书的内容规范

## Input:
- 用户的创作需求：```{history}```
- 文案属性（风格、受众等）: ```{scene_attributes}```
- 参考文案库内容：```{rag_context}```
- 实时搜索结果：```{search_context}```
"""


RAG_SCENE_MAP = {
    "1": ("生活分享", "1-lifestyle"),
    "2": ("美妆护肤", "2-beauty"),
    "3": ("时尚穿搭", "3-fashion"),
    "4": ("运动健康", "4-fitness"),
    "5": ("科技数码", "5-tech"),
    "6": ("音乐影视", "6-entertainment"),
    "7": ("书籍阅读", "7-reading"),
    "8": ("宠物生活", "8-pets"),
}


def get_llm(model:Optional[str] = None):
    return ZhipuLLM(model)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return SiliconFlowEmbeddings()


def format_history(history: List[Dict[str, str]]) -> str:
    return str(history)


def _sanitize_json(text: str) -> str:
    return (
        text.replace("```json", "")
        .replace("```", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("，", ",")
        .strip()
    )


def ensure_scene_attributes(scene_label: str, current: Dict[str, str]) -> Dict[str, str]:
    if scene_label and current:
        return current
    scene, attrs, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    if not attrs:
        return current
    return {attr: current.get(attr, "") for attr in attrs}


def build_retrievers(chunk_size: int = 896, force: bool = False):
    embeddings = get_embeddings()
    retrievers: Dict[str, Any] = {}
    dest = os.path.join(TIANJI_PATH, "temp", "tianji-chinese")
    if not os.path.exists(dest):
        # 复用 demo_rag_langchain_all.py 的下载逻辑
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


RAG_RETRIEVERS = build_retrievers()
LLM = get_llm()
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_CLIENT = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None


def llm_invoke(prompt: str):
    if hasattr(LLM, "invoke"):
        return LLM.invoke(prompt)
    if hasattr(LLM, "_call"):
        return LLM._call(prompt)
    return LLM(prompt)


# --------------- LangGraph 节点实现 ---------------
def intent_node(state: TianjiState):
    prompt = INTENT_PROMPT_TEMPLATE.format(
        instruction=format_history(state["chat_history"]),
        scene=SCENE_OPTIONS,
        scene_example=SCENE_EXAMPLES,
    )
    rsp = llm_invoke(prompt)
    intent = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
    return {"scene_label": intent.strip()}


def refine_node(state: TianjiState):
    scene_label_raw = state.get("scene_label", "").strip()
    scene_label = scene_label_raw.split("：")[0]
    if not scene_label or scene_label == "None" or not scene_label.isdigit() or not is_number_in_types(
        SCENE_JSON, int(scene_label)
    ):
        st.warning("此模型只支持回答关于人情世故的事项，已调用 API 为你进行单轮回答。")
        rsp = llm_invoke(prompt=state["user_input"])
        return {"need_more_info": True, "final_answer": rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))}

    base_attrs = ensure_scene_attributes(scene_label, state.get("scene_attributes", {}))
    scene, _, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    desc = extract_attribute_descriptions(SCENE_JSON, base_attrs)

    refine_prompt = REFINE_PROMPT_TEMPLATE.format(
        instruction=format_history(state["chat_history"]),
        scene=scene,
        scene_attributes=base_attrs,
        scene_attributes_description=desc,
    )
    refined_text = llm_invoke(refine_prompt)
    refined = refined_text if isinstance(refined_text, str) else getattr(refined_text, "content", "")
    merged = base_attrs
    try:
        parsed = json.loads(_sanitize_json(refined))
        merged = {**base_attrs, **parsed}
    except Exception:
        merged = base_attrs

    if has_empty_values(merged):
        question_prompt = QUESTION_PROMPT_TEMPLATE.format(
            scene=scene,
            scene_attributes=merged,
            scene_attributes_description=desc,
        )
        question = llm_invoke(question_prompt)
        q_content = question if isinstance(question, str) else getattr(question, "content", "")
        return {
            "scene_attributes": merged,
            "need_more_info": q_content.strip() != "Full",
            "final_answer": q_content if q_content.strip() != "Full" else "",
        }

    return {"scene_attributes": merged, "need_more_info": False}


def rag_node(state: TianjiState):
    scene_label = state.get("scene_label", "").split("：")[0].strip()
    retriever = RAG_RETRIEVERS.get(scene_label)
    docs = []
    if retriever:
        try:
            docs = retriever.invoke(state["user_input"]) or []
        except Exception:
            docs = []
    doc_texts = [d.page_content for d in docs][:5]
    return {"retrieved_docs": doc_texts}


def _generate_queries(state: TianjiState) -> List[str]:
    attrs = state.get("scene_attributes", {})
    base = state.get("user_input", "")
    scene_label = state.get("scene_label", "")
    queries = [base]
    extra_bits = [v for v in attrs.values() if v]
    if scene_label:
        queries.append(f"场景{scene_label} {base}")
    if extra_bits:
        queries.append(base + " " + " ".join(extra_bits[:3]))
    return queries


def search_node(state: TianjiState):
    if not TAVILY_CLIENT:
        return {"search_results": {}}
    queries = _generate_queries(state)
    results: Dict[str, Any] = {}
    for idx, q in enumerate(queries):
        try:
            resp = TAVILY_CLIENT.search(q, max_results=5)
            results[str(idx)] = resp.get("results", [])
        except Exception:
            results[str(idx)] = []
    return {"search_results": results}


def _format_rag_docs(docs: List[str]) -> str:
    if not docs:
        return ""
    return "\n\n".join(docs[:5])


def _format_search_results(results: Dict[str, Any]) -> str:
    if not results:
        return ""
    lines = []
    for _, items in results.items():
        for item in items:
            url = item.get("url") or item.get("link")
            content = item.get("content") or item.get("snippet") or ""
            title = item.get("title", "")
            lines.append(f"{title}\n{url}\n{content}")
    return "\n\n".join(lines[:5])


def answer_node(state: TianjiState):
    scene_label = state.get("scene_label", "").split("：")[0].strip()
    scene, _, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    rag_ctx = _format_rag_docs(state.get("retrieved_docs", []))
    search_ctx = _format_search_results(state.get("search_results", {}))
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        scene=scene,
        scene_attributes=state.get("scene_attributes", {}),
        rag_context=rag_ctx,
        search_context=search_ctx,
        history=format_history(state["chat_history"]),
    )
    rsp = llm_invoke(prompt)
    ans = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
    st.session_state["chat_completed"] = True
    return {"final_answer": ans}


# --------------- LangGraph 构建 ---------------
@st.cache_resource(show_spinner=False)
def build_app():
    workflow = StateGraph(TianjiState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("intent")
    workflow.add_edge("intent", "refine")

    def after_refine(state: TianjiState):
        return END if state.get("need_more_info") else "rag"

    workflow.add_conditional_edges("refine", after_refine, {"rag": "rag", END: END})

    def after_rag(state: TianjiState):
        return "search" if state.get("search_enabled") else "answer"

    workflow.add_conditional_edges("rag", after_rag, {"search": "search", "answer": "answer"})
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


APP = build_app()

# --------------- Streamlit UI ---------------
st.set_page_config(page_title="小红书文案生成器", page_icon="✨")

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
    st.session_state["chat_history"] = []
    st.session_state["scene_label"] = ""
    st.session_state["scene_attributes"] = {}


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

st.title("✨ 小红书智能文案生成器")

for idx, turn in enumerate(st.session_state["chat_history"]):
    if turn["role"] == "user":
        message(turn["content"], is_user=True, key=f"user_{idx}")
    else:
        message(turn["content"], is_user=False, key=f"assistant_{idx}")

if user_input := st.chat_input("💡 告诉我你想创作什么样的文案..."):
    logger.info(f"用户输入：{user_input}")
    logger.info(f"历史对话：{st.session_state['chat_history']}")
        # 如果上一个对话已完成，清空所有数据开启新对话
    if st.session_state.get("chat_completed", False):
        st.session_state["chat_history"] = []
        st.session_state["scene_label"] = ""
        st.session_state["scene_attributes"] = {}
        st.session_state["chat_completed"] = False

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    init_state: TianjiState = {
        "user_input": user_input,
        "chat_history": st.session_state["chat_history"],
        "scene_label": st.session_state.get("scene_label", ""),
        "scene_attributes": st.session_state.get("scene_attributes", {}),
        "retrieved_docs": [],
        "search_enabled": st.session_state.get("enable_se", False),
        "search_results": {},
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