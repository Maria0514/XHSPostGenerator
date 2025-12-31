"""LangGraph 工作流构建"""

from typing import Any, Dict
from langgraph.graph import END, StateGraph
from .nodes import *
from .spoilState import SpoilState
from spoil.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM, ZhipuLLM



def build_xhs_workflow(retrievers: Dict[str, Any], tavily_client: Any = None):
    """
    构建小红书文案生成工作流
    
    Args:
        llm: 语言模型实例
        retrievers: 各场景的检索器字典
        tavily_client: Tavily 搜索客户端（可选）
    
    Returns:
        编译后的 LangGraph 工作流
    """
    workflow = StateGraph(SpoilState)

    # 不同节点模型配置
    intent_llm = SiliconFlowLLM(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')
    refine_llm = SiliconFlowLLM(model='deepseek-ai/DeepSeek-V3.2')
    question_llm = SiliconFlowLLM(model='deepseek-ai/DeepSeek-V3.2')
    extend_query_llm = SiliconFlowLLM(model='deepseek-ai/DeepSeek-V3.2')
    fillter_web_llm = SiliconFlowLLM(model='deepseek-ai/DeepSeek-V3.2')
    answer_llm = SiliconFlowLLM(model='deepseek-ai/DeepSeek-V3.2') # todo:使用微调模型

    # 添加节点
    workflow.add_node("intent", lambda state: intent_node(state, intent_llm))
    workflow.add_node("refine", lambda state: refine_node(state, refine_llm))
    workflow.add_node("rag", lambda state: rag_node(state, retrievers))
    workflow.add_node("question", lambda state: question_node(state, question_llm))
    workflow.add_node("extend_query", lambda state: extend_query_node(state, extend_query_llm))
    workflow.add_node("search", lambda state: search_node(state, tavily_client))
    workflow.add_node("fillter_web", lambda state: fillter_web_node(state, fillter_web_llm))
    workflow.add_node("answer", lambda state: answer_node(state, answer_llm))

    
    # 设置入口点
    workflow.set_entry_point("intent")
    workflow.add_edge("intent", "refine")
    
    # 条件分支：refine 之后，若需要追问则进入 question，否则进行 rag
    def after_refine(state: SpoilState):
        return "question" if state.get("need_more_info") else "rag"
    
    workflow.add_conditional_edges("refine", after_refine, {"rag": "rag", "question": "question"})
    workflow.add_edge("question", END)
    
    # 条件分支：rag 之后，若开启了搜索功能则进行：扩展查询->搜索->过滤网页；否则直接生成答案
    def after_rag(state: SpoilState):
        return "extend_query" if state.get("search_enabled") else "answer"
    
    workflow.add_conditional_edges("rag", after_rag, {"extend_query": "extend_query", "answer": "answer"})
    
    # 其他边
    workflow.add_edge("extend_query", "search")
    workflow.add_edge("search", "fillter_web")
    workflow.add_edge("fillter_web", "answer")
    workflow.add_edge("answer", END)
    
    return workflow.compile()