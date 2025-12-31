"""意图识别 Node"""

from typing import Any, Dict, List, TypedDict
from ..config import SCENE_JSON, SCENE_OPTIONS, SCENE_EXAMPLES
from ..prompts import INTENT_PROMPT_TEMPLATE
from ..spoilState import SpoilState
from loguru import logger

def format_history(history: List[Dict[str, str]]) -> str:
    return str(history)


def llm_invoke(prompt: str, llm: Any) -> str:
    """调用 LLM"""
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "_call"):
        return llm._call(prompt)
    return llm(prompt)


def intent_node(state: SpoilState, llm: Any):
    """
    意图识别节点：根据用户输入识别内容类型
    
    Args:
        state: 工作流状态
        llm: 语言模型实例
    
    Returns:
        更新后的状态字典，包含识别到的 scene_label
    """
    prompt = INTENT_PROMPT_TEMPLATE.format(
        instruction=format_history(state["chat_history"]),
        scene=SCENE_OPTIONS,
        scene_example=SCENE_EXAMPLES,
    )
    rsp = llm_invoke(prompt, llm)
    logger.info(f"意图识别结果：{rsp}，当前对话历史: {state['chat_history']}")
    intent = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
    return {"scene_label": intent.strip()}