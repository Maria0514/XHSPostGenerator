"""
Streamlit + LangGraph 版本的天机智能体
全部改为 LangChain/LangGraph，实现 RAG（参考 demo_rag_langchain_all.py）与可选联网搜索。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from tianji.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM


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
#Role:
- 场景分析助手

## Background:
- 作为一个专业的场景分析助手。接下来，我将向你展示一段用户与大模型的历史对话记录，user 表示用户，assistant 表示大模型，你需要从中判断对话属于哪个场景。

## Goals:
- 你的任务是准确判断最新的用户提问符合哪个场景，用户身处在哪个场景，用户想要大模型提供哪种场景下的帮助。

## Constraints:
- 你只需要用代表场景标签的数字回复（例如场景标签是"4：送祝福"，则回复数字 "4"），不需要回复其他任何内容！
- 你需要根据历史对话记录判断用户的场景是否发生改变，如果是，回复最新的场景即可。
- 如果历史对话都不符合场景标签选项，请只返回字符串"None"。
- 你无需输出思考过程，直接返回答案即可。

## Inputs:
- 历史对话记录：```{instruction}```
- 场景标签选项: ```{scene}```
- 关于场景标签选项的细分场景:```{scene_example}```
"""

REFINE_PROMPT_TEMPLATE = """
#Role:
- 场景细化小助手

## Background:
- 作为一个专业的{scene}场景分析助手。接下来，我将向你展示一段用户与大模型的历史对话记录，user 表示用户，assistant 表示大模型，你需要从中提取相对应的场景要素并组装成json。

## Goals:
- 我将提供给你需要提取的场景要素，你的任务是从历史对话记录中的内容分析并提取对应场景的场景要素。

## Constraints:
- 只返回单个 json 对象，不要返回其他内容。
- 如果没有提取到对应的场景要素请用空字符串表示，例如："对象角色": ""。
- 如果发现场景要素发生更新，覆盖旧值。

## Input:
- 历史对话记录：```{instruction}```
- 需要提取的场景要素: ```{scene_attributes}```
- 每个场景要素的描述以及例子:```{scene_attributes_description}```
"""

QUESTION_PROMPT_TEMPLATE = """
#Role:
- 提问小助手

## Goals:
- 给出针对空缺场景要素的单个追问。

## Constraints:
- 如果所有场景要素都有值，回复字符串"Full"。
- 只问一个问题。

## Input:
- 用户面对的场景：```{scene}```
- 当前场景要素: ```{scene_attributes}```
- 每个场景要素的描述以及例子:```{scene_attributes_description}```
"""

ANSWER_PROMPT_TEMPLATE = """
#Role:
- {scene}小助手

## Goals:
- 根据场景要素、检索到的 RAG 片段和搜索结果，结合历史对话，给出定制化回答。

## Constraints:
- 需要基于提供的场景要素与上下文进行详细回答，避免泛泛而谈。
- 如果搜索结果不为空，优先基于搜索内容；若为空，则结合 RAG 结果；都为空再用常识补充。

## Input:
- 历史对话记录：```{history}```
- 场景要素: ```{scene_attributes}```
- RAG 上下文：```{rag_context}```
- 搜索结果：```{search_context}```
"""


RAG_SCENE_MAP = {
    "1": ("敬酒礼仪文化", "1-etiquette"),
    "2": ("请客礼仪文化", "2-hospitality"),
    "3": ("送礼礼仪文化", "3-gifting"),
    "4": ("送祝福", None),
    "5": ("如何说对话", "5-communication"),
    "6": ("化解尴尬场合", "6-awkwardness"),
    "7": ("矛盾与冲突应对", "7-conflict"),
}


@st.cache_resource(show_spinner=False)
def get_llm():
    return SiliconFlowLLM()


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
        return {"need_more_info": True, "final_answer": "该问题不在支持的场景内，请换个提问。"}

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
    return {"final_answer": ans}


# --------------- LangGraph 构建 ---------------
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
st.set_page_config(page_title="天机 LangGraph", page_icon="🤖")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "scene_label" not in st.session_state:
    st.session_state["scene_label"] = ""
if "scene_attributes" not in st.session_state:
    st.session_state["scene_attributes"] = {}
if "enable_se" not in st.session_state:
    st.session_state["enable_se"] = False


def reset_chat():
    st.session_state["chat_history"] = []
    st.session_state["scene_label"] = ""
    st.session_state["scene_attributes"] = {}


with st.sidebar:
    st.markdown("## 支持场景")
    for item in SCENE_OPTIONS:
        st.write(item)
    st.checkbox("启用网络搜索（需要 TAVILY_API_KEY）", key="enable_se")
    st.button("清空对话", on_click=reset_chat)

st.title("人情世故大模型 · LangGraph 版")

for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if user_input := st.chat_input("请输入问题"):
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

    result = APP.invoke(init_state)

    if result.get("scene_label"):
        st.session_state["scene_label"] = result["scene_label"]
    if result.get("scene_attributes"):
        st.session_state["scene_attributes"] = result["scene_attributes"]

    assistant_text = result.get("final_answer", "")
    if assistant_text:
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

    if result.get("need_more_info") and not assistant_text:
        # 没有问题文本时给出兜底提示
        fallback = "我需要更多场景要素，请补充信息。"
        st.session_state["chat_history"].append({"role": "assistant", "content": fallback})
        with st.chat_message("assistant"):
            st.markdown(fallback)
