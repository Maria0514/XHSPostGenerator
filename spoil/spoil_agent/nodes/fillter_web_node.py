"""网页内容过滤和爬取 Node

基于大模型的选择，并发爬取最有价值的网页，提取关键信息。
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from loguru import logger

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

from ..prompts import SELECT_WEBPAGE_PROMPT_TEMPLATE
from ..spoilState import SpoilState


# 反爬虫网站黑名单
BLOCKED_DOMAINS = {
    "zhihu.com",
    "baidu.com",
    "sohu.com",
    "qq.com",
    "360.cn",
    "bing.com",
    "sogou.com",
    "sm.cn",
}


def llm_invoke(prompt: str, llm: Any) -> str:
    """调用 LLM"""
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "_call"):
        return llm._call(prompt)
    return llm(prompt)


def _is_blocked_domain(url: str) -> bool:
    """检查是否是黑名单域名"""
    try:
        domain = urlparse(url).netloc.lower()
        # 移除 www. 前缀
        domain = domain.replace("www.", "")
        return any(blocked in domain for blocked in BLOCKED_DOMAINS)
    except Exception:
        return False


def _fetch_webpage_content(url: str, timeout: int = 5) -> Optional[str]:
    """爬取网页内容"""
    if not requests or not BeautifulSoup:
        return None
    
    if _is_blocked_domain(url):
        return None
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.encoding = response.apparent_encoding or "utf-8"
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 提取主要内容
        # 优先查找 main、article、content 等标签
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_=lambda x: x and "content" in x.lower())
            or soup.find(class_=lambda x: x and "article" in x.lower())
            or soup.body
        )
        
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        
        # 清理过多的空行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)
        
        # 限制长度
        if len(text) > 2000:
            text = text[:2000] + "..."
        logger.info(f"爬取到网页内容：{text}")
        return text if text else None
    except Exception:
        return None


def _select_webpages(
    state: SpoilState,
    search_results: Dict[str, Any],
    llm: Any,
) -> List[Dict[str, str]]:
    """使用大模型选择最有价值的网页"""
    
    # 整理搜索结果
    formatted_results = []
    seen_urls = set()
    
    for _, lst in search_results.items():
        if not isinstance(lst, list):
            continue
        for item in lst:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or item.get("link") or "").strip()
            if not url or url in seen_urls or _is_blocked_domain(url):
                continue
            title = (item.get("title") or "").strip()
            snippet = (item.get("content") or item.get("snippet") or "").strip()
            seen_urls.add(url)
            formatted_results.append(
                {"url": url, "title": title, "snippet": snippet}
            )
    
    if not formatted_results:
        return []
    
    # 构建提示词
    prompt = SELECT_WEBPAGE_PROMPT_TEMPLATE.format(
        user_input=state.get("user_input", ""),
        scene_attributes=str(state.get("scene_attributes", {})),
        search_results=str(formatted_results[:10]),  # 只传前 10 个
    )
    
    try:
        rsp = llm_invoke(prompt, llm)
        text = rsp if isinstance(rsp, str) else getattr(rsp, "content", str(rsp))
        
        # 清理响应
        cleaned = (
            text.replace("```", "")
            .replace("```json", "")
            .replace(""", '"')
            .replace(""", '"')
            .strip()
        )
        logger.info(f"选择最有价值的网页结果：{cleaned}")
        selected = ast.literal_eval(cleaned)
        if not isinstance(selected, list):
            return []
        
        # 验证格式
        valid_selected = []
        for item in selected:
            if isinstance(item, dict) and "url" in item:
                valid_selected.append(item)
        
        return valid_selected[:5]  # 最多 5 个
    except Exception:
        # 如果 LLM 失败，返回空列表（使用备选方案）
        return []


def _fallback_webpages(search_results: Dict[str, Any]) -> List[Dict[str, str]]:
    """备选方案：自动选择前几个非黑名单网页"""
    selected = []
    seen_urls = set()
    
    for _, lst in search_results.items():
        if not isinstance(lst, list):
            continue
        for item in lst:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or item.get("link") or "").strip()
            if not url or url in seen_urls or _is_blocked_domain(url):
                continue
            seen_urls.add(url)
            selected.append({"url": url, "reason": "auto-selected"})
            if len(selected) >= 5:
                return selected
    
    return selected


def _fetch_and_format_webpage(item: Dict[str, str]) -> Optional[str]:
    """爬取并格式化单个网页"""
    url = item.get("url", "").strip()
    if not url:
        return None
    
    content = _fetch_webpage_content(url)
    if content:
        reason = item.get("reason", "")
        title = item.get("title", "")
        return f"【{title or url}】\n{reason}\n\n{content}"
    return None


def fillter_web_node(state: SpoilState, llm: Any = None):
    """
    网页过滤和内容提取节点。
    
    1. 使用大模型选择最有价值的网页
    2. 并发爬取选中网页的完整内容
    3. 整理成可用于 LLM 的 search_context
    """
    
    raw: Dict[str, Any] = state.get("search_results") or {}
    if not raw:
        return {"search_context": ""}
    
    # 使用大模型选择网页（如果可用）
    if llm:
        selected_webpages = _select_webpages(state, raw, llm)
    else:
        selected_webpages = []
    
    # 备选方案：如果大模型选择失败，使用自动选择
    if not selected_webpages:
        selected_webpages = _fallback_webpages(raw)
    
    if not selected_webpages:
        return {"search_context": ""}
    
    # 使用线程池并发爬取网页内容
    contents: List[str] = []
    max_workers = min(5, len(selected_webpages))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有爬取任务
        future_to_item = {
            executor.submit(_fetch_and_format_webpage, item): item
            for item in selected_webpages
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_item):
            try:
                result = future.result()
                if result:
                    contents.append(result)
            except Exception:
                pass
    
    if not contents:
        return {"search_context": ""}
    
    # 整理成最终的 search_context
    search_context = "\n\n---\n\n".join(contents)
    
    # 限制总长度
    if len(search_context) > 5000:
        search_context = search_context[:5000] + "\n\n[内容已截断...]"
    
    return {"search_context": search_context}