"""扩展查询 Node

该节点基于用户的完整对话历史和已提取的属性，生成 3-5 个高质量的搜索查询。
这些查询用于验证信息的真实性和获取最新的、具体的参考内容。
优先用 LLM 生成结构化列表；失败则回退到规则拼接。
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List

from spoil.agents.metagpt_agents.utils.helper_func import (
    extract_single_type_attributes_and_examples,
)
from ..config import SCENE_JSON
from ..prompts import EXTEND_QUERY_PROMPT_TEMPLATE
from ..spoilState import SpoilState
from loguru import logger


def llm_invoke(prompt: str, llm: Any) -> str:
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "_call"):
        return llm._call(prompt)
    return llm(prompt)


def _fallback_queries(state: SpoilState) -> List[str]:
    """基于用户输入和属性生成回退查询"""
    base = (state.get("user_input") or "").strip()
    scene_label = (state.get("scene_label") or "").strip()
    attrs = state.get("scene_attributes") or {}

    # 从属性中提取关键信息
    extra_bits = [v for v in attrs.values() if isinstance(v, str) and v.strip()]
    
    queries: List[str] = []
    
    # 基础查询：用户输入
    if base:
        queries.append(base)
    
    # 属性驱动的查询：结合具体属性
    if extra_bits:
        # 优先搜索具体的实体信息
        queries.append(" ".join(extra_bits[:2]))
        
        # 搜索组合：用户需求 + 属性
        if base and extra_bits:
            queries.append(base + " " + " ".join(extra_bits[:2]))
        
        # 搜索组合：属性 + 验证性关键词
        queries.append(" ".join(extra_bits[:3]) + " 真实")
    
    # 通用查询：小红书相关内容
    if base or extra_bits:
        combined = base + " " + " ".join(extra_bits[:2])
        queries.append("小红书 " + combined.strip())

    # 去重保持顺序
    dedup: List[str] = []
    seen = set()
    for q in queries:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        dedup.append(q)
    return dedup[:5]


def extend_query_node(state: SpoilState, llm: Any):
    """
    生成搜索查询列表。
    
    基于用户的完整对话历史、已提取的属性和场景信息，生成高质量的搜索查询。
    这些查询用于验证信息真实性和获取最新的参考内容。
    """

    base = (state.get("user_input") or "").strip()
    attrs = state.get("scene_attributes") or {}
    scene_label = (state.get("scene_label") or "").strip()
    chat_history = state.get("chat_history") or []

    # 低信息输入直接回退
    if not base:
        return {"search_queries": []}

    # 获取场景名称
    scene = ""
    if scene_label:
        try:
            scene, _, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
        except Exception:
            scene = ""

    # 使用模板生成提示词
    prompt = EXTEND_QUERY_PROMPT_TEMPLATE.format(
        chat_history=str(chat_history),
        user_input=base,
        scene_attributes=str(attrs),
        scene=scene or "通用",
    )

    try:
        rsp = llm_invoke(prompt, llm)
        text = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
        cleaned = (
            text.replace("```", "")
            .replace("```list", "")
            .replace("，", ",")
            .strip()
        )
        queries = ast.literal_eval(cleaned)
        if not isinstance(queries, list):
            raise ValueError("LLM did not return a list")
        queries = [str(q).strip() for q in queries if str(q).strip()]
        # 去重
        dedup: List[str] = []
        seen = set()
        for q in queries:
            if q in seen:
                continue
            seen.add(q)
            dedup.append(q)
        logger.info(f"额外生成的查询：{dedup}")
        return {"search_queries": dedup[:5]}
    except Exception:
        return {"search_queries": _fallback_queries(state)}