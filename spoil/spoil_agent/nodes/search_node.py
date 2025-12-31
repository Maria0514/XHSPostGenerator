"""网络搜索 Node"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, TypedDict
from ..spoilState import SpoilState



def _generate_queries(state: SpoilState) -> List[str]:
    """生成搜索查询"""
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


def _search_query(tavily_client: Any, query: str) -> Dict[str, Any]:
    """执行单个搜索查询"""
    try:
        resp = tavily_client.search(query, max_results=5)
        return resp.get("results", [])
    except Exception:
        return []


def search_node(state: SpoilState, tavily_client: Any):
    """
    网络搜索节点：使用 Tavily 并发搜索实时信息
    
    Args:
        state: 工作流状态
        tavily_client: Tavily 客户端
    
    Returns:
        更新后的状态字典，包含搜索结果
    """
    if not tavily_client:
        return {"search_results": {}}
    
    queries = state.get("search_queries") or _generate_queries(state)
    if not queries:
        return {"search_results": {}}
    
    results: Dict[str, Any] = {}
    
    # 使用线程池并发执行搜索
    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
        # 提交所有搜索任务
        future_to_idx = {
            executor.submit(_search_query, tavily_client, q): idx
            for idx, q in enumerate(queries)
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                search_results = future.result()
                results[str(idx)] = search_results
            except Exception:
                results[str(idx)] = []
    
    return {"search_results": results}