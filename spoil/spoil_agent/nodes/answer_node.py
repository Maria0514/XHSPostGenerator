"""文案生成 Node"""

from typing import Any, Dict, List, TypedDict

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from spoil.agents.metagpt_agents.utils.helper_func import (
    extract_single_type_attributes_and_examples,
)
from ..config import SCENE_JSON
from ..prompts import ANSWER_PROMPT_TEMPLATE
from ..spoilState import SpoilState


def format_history(history: List[Dict[str, str]]) -> str:
    return str(history)


def llm_invoke(prompt: str, llm: Any) -> str:
    """调用 LLM"""
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "_call"):
        return llm._call(prompt)
    return llm(prompt)


def _format_rag_docs(docs: List[str]) -> str:
    """格式化 RAG 文档"""
    if not docs:
        return ""
    return "\n\n".join(docs[:5])


def _format_search_results(results: Dict[str, Any]) -> str:
    """格式化搜索结果"""
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


def answer_node(state: SpoilState, llm: Any):
    """
    文案生成节点：基于所有信息生成最终的小红书文案
    
    Args:
        state: 工作流状态
        llm: 语言模型实例
    
    Returns:
        更新后的状态字典，包含生成的文案
    """
    scene_label = state.get("scene_label", "").split("：")[0].strip()
    scene, _, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    rag_ctx = _format_rag_docs(state.get("retrieved_docs", []))
    search_ctx = state.get("search_context") or _format_search_results(state.get("search_results", {}))
    
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        scene=scene,
        scene_attributes=state.get("scene_attributes", {}),
        rag_context=rag_ctx,
        search_context=search_ctx,
        history=format_history(state["chat_history"]),
    )
    
    rsp = llm_invoke(prompt, llm)
    ans = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
    
    return {"final_answer": ans, "chat_completed": True}