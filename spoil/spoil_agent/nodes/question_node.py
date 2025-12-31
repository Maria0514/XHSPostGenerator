"""追问 Node

当 refine_node 判定还缺关键属性时，用该节点生成一个自然的追问。
"""

from __future__ import annotations

from typing import Any

from spoil.agents.metagpt_agents.utils.helper_func import (
    extract_single_type_attributes_and_examples,
    extract_attribute_descriptions,
)

from ..config import SCENE_JSON
from ..prompts import QUESTION_PROMPT_TEMPLATE
from ..spoilState import SpoilState


def llm_invoke(prompt: str, llm: Any) -> str:
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "_call"):
        return llm._call(prompt)
    return llm(prompt)


def question_node(state: SpoilState, llm: Any):
    # refine_node 可能已经生成了 final_answer（追问）。这里做兜底。
    if state.get("final_answer"):
        # 更新 chat_history，将追问添加到对话历史
        final_answer = state.get("final_answer", "")
        updated_history = state.get("chat_history", []) + [
            {"role": "assistant", "content": final_answer}
        ]
        return {
            "final_answer": final_answer,
            "need_more_info": True,
            "chat_history": updated_history
        }

    scene_label = (state.get("scene_label") or "").split("：")[0].strip()
    scene, _, _ = extract_single_type_attributes_and_examples(SCENE_JSON, scene_label)
    attrs = state.get("scene_attributes") or {}
    desc = extract_attribute_descriptions(SCENE_JSON, attrs)

    prompt = QUESTION_PROMPT_TEMPLATE.format(
        scene=scene,
        scene_attributes=attrs,
        scene_attributes_description=desc,
    )
    rsp = llm_invoke(prompt, llm)
    text = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
    if text.strip() == "Full":
        return {"need_more_info": False}
    
    # 追问时也要更新 chat_history
    updated_history = state.get("chat_history", []) + [
        {"role": "assistant", "content": text}
    ]
    return {
        "final_answer": text,
        "need_more_info": True,
        "chat_history": updated_history
    }