"""RAG 检索 Node"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..spoilState import SpoilState


def _collect_user_messages(state: SpoilState, *, min_len: int) -> List[str]:
    """收集对话历史中所有“足够长”的用户消息（去重保序）。

    用于多轮追问场景：最新一轮用户输入往往只是属性补充（信息量小），
    因此将所有长度>=min_len 的用户输入拼起来作为检索 query，更稳定。
    """
    history = state.get("chat_history") or []
    messages: List[str] = []
    seen = set()
    for turn in history:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if len(content) < min_len:
            continue
        if content in seen:
            continue
        seen.add(content)
        messages.append(content)
    return messages


def _build_rag_query(state: SpoilState, *, min_user_input_len: int = 6, max_attrs: int = 6) -> str:
    """构造更“懂需求”的检索 query：用户需求 + 已确认属性（键值对）。"""
    user_msgs = _collect_user_messages(state, min_len=min_user_input_len)
    user_query = "\n".join(user_msgs).strip()
    if not user_query:
        # 极端情况下（历史里没有足够长的用户输入）再回退到当前输入
        user_query = (state.get("user_input") or "").strip()

    attrs = state.get("scene_attributes") or {}

    # 只拼接非空属性，并做去重与长度控制，避免 query 过长/过噪。
    kv_pairs: List[str] = []
    seen_values = set()
    for key, value in attrs.items():
        if len(kv_pairs) >= max_attrs:
            break
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        if value in seen_values:
            continue
        seen_values.add(value)
        kv_pairs.append(f"{key}={value}")

    if user_query and kv_pairs:
        return f"{user_query}\n属性：" + "；".join(kv_pairs)
    return user_query


def rag_node(state: SpoilState, retrievers: Dict[str, Any]):
    """
    RAG 检索节点：从知识库中检索相关文案
    
    Args:
        state: 工作流状态
        retrievers: 各场景的检索器字典
    
    Returns:
        更新后的状态字典，包含检索到的文档
    """
    scene_label = state.get("scene_label", "").split("：")[0].strip()
    retriever = retrievers.get(scene_label)
    docs = []
    
    if retriever:
        try:
            query = _build_rag_query(state)
            docs = retriever.invoke(query) or []
        except Exception:
            docs = []
    
    doc_texts = [d.page_content for d in docs][:3]
    return {"retrieved_docs": doc_texts}