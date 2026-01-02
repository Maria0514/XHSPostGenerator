from typing import Any, Dict, List, TypedDict

class SpoilState(TypedDict):
    user_input: str
    chat_history: List[Dict[str, str]]
    scene_label: str
    scene_attributes: Dict[str, str]
    retrieved_docs: List[str]
    search_enabled: bool
    # 由 extend_query_node 生成，用于 search_node
    search_queries: List[str]
    search_results: Dict[str, Any]
    # 由 fillter_web_node 生成，给 answer_node 拼接上下文
    search_context: str
    final_answer: str
    need_more_info: bool
    # 标记是否已输出最终答案（用于 UI 侧控制是否开启新对话）
    chat_completed: bool