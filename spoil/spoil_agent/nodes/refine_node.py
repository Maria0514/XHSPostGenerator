"""属性提取 Node"""

import json
from typing import Any, Dict, List

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from spoil.agents.metagpt_agents.utils.helper_func import (
    extract_single_type_attributes_and_examples,
    extract_attribute_descriptions,
    has_empty_values,
    is_number_in_types,
)
from ..config import SCENE_JSON
from ..prompts import REFINE_PROMPT_TEMPLATE
from ..spoilState import SpoilState


def format_history(history: List[Dict[str, str]]) -> str:
    return str(history)


def _sanitize_json(text: str) -> str:
    """清理 JSON 文本"""
    return (
        text.replace("```json", "")
        .replace("```", "")
        .replace("\"", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("，", ",")
        .strip()
    )


def llm_invoke(prompt: str, llm: Any) -> str:
    """调用 LLM"""
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "_call"):
        return llm._call(prompt)
    return llm(prompt)


def ensure_scene_attributes(scene_label: str, current: Dict[str, str]) -> Dict[str, str]:
    """确保场景属性完整"""
    if scene_label and current:
        return current
    scene, attrs, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    if not attrs:
        return current
    return {attr: current.get(attr, "") for attr in attrs}


def refine_node(state: SpoilState, llm: Any):
    """
    属性提取节点：从用户输入中提取文案创作所需的属性
    
    Args:
        state: 工作流状态
        llm: 语言模型实例
    
    Returns:
        更新后的状态字典，包含提取的属性和是否需要更多信息
    """
    scene_label_raw = state.get("scene_label", "").strip()
    scene_label = scene_label_raw.split("：")[0]
    
    if not scene_label or scene_label == "None" or not scene_label.isdigit() or not is_number_in_types(
        SCENE_JSON, int(scene_label)
    ):
        if st is not None:
            try:
                st.warning("此模型只支持回答关于小红书文案创作的事项，已调用 API 为你进行单轮回答。")
            except Exception:
                pass
        rsp = llm_invoke(state["user_input"], llm=llm)
        return {"need_more_info": True, "final_answer": rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))}

    base_attrs = ensure_scene_attributes(scene_label, state.get("scene_attributes", {}))
    scene, _, example = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    desc = extract_attribute_descriptions(SCENE_JSON, base_attrs)

    refine_prompt = REFINE_PROMPT_TEMPLATE.format(
        instruction=format_history(state["chat_history"]),
        scene=scene,
        scene_attributes=base_attrs,
        scene_attributes_description=desc,
    )
    refined_text = llm_invoke(refine_prompt, llm)
    refined = refined_text if isinstance(refined_text, str) else getattr(refined_text, "content", "")
    merged = base_attrs
    
    try:
        parsed = json.loads(_sanitize_json(refined))
        merged = {**base_attrs, **parsed}
    except Exception:
        merged = base_attrs

    # 如果属性有空值，标记需要更多信息，由 question_node 专门处理
    need_more_info = has_empty_values(merged)
    return {"scene_attributes": merged, "need_more_info": need_more_info}