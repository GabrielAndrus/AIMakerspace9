"""Doc Fetcher Agent - Prioritizes and fetches URLs for documentation."""

import requests
import re

from src.agent.error_investigator import fetch_documentation


def doc_fetcher_node(state: dict) -> dict:
    """LangGraph node that fetches documentation from URLs.
    
    Input state keys used: search_results, relevant_urls (optional)
    Output state keys set: fetched_docs
    
    Uses the existing fetch_documentation tool or similar pattern.
    Prioritizes URLs based on source reliability and relevance.
    
    Args:
        state: Current graph state containing search results
        
    Returns:
        Partial state dict with fetched documentation
    """
    search_results = state.get("search_results", [])
    relevant_urls = state.get("relevant_urls", [])
    
    if not search_results and not relevant_urls:
        return {"fetched_docs": []}
    
    urls_to_fetch = _prioritize_urls(search_results, relevant_urls)
    
    fetched_docs = []
    
    for url in urls_to_fetch[:3]:
        try:
            print(f"[DocFetcher] Fetching: {url[:60]}...")
            
            content = fetch_documentation.invoke({"url": url})
            
            if content and len(content) > 100:
                source_name = _get_source_name(url)
                
                fetched_docs.append({
                    "url": url,
                    "source": source_name,
                    "content": content[:6000],
                    "fetch_success": True
                })
            else:
                fetched_docs.append({
                    "url": url,
                    "source": _get_source_name(url),
                    "content": "",
                    "fetch_success": False,
                    "error": "Content too short or empty"
                })
                
        except Exception as e:
            print(f"[DocFetcher] Error fetching {url}: {e}")
            fetched_docs.append({
                "url": url,
                "source": _get_source_name(url),
                "content": "",
                "fetch_success": False,
                "error": str(e)[:100]
            })
    
    return {"fetched_docs": fetched_docs}


def _prioritize_urls(search_results: list[dict], relevant_urls: list[str]) -> list[str]:
    """Prioritize URLs for fetching based on source reliability.
    
    Priority order:
    1. GitHub issues (often have exact error + fix discussion)
    2. Hugging Face docs (authoritative for transformers/TRL)
    3. Stack Overflow (verified solutions)
    4. Reddit/discussions (contextual understanding)
    5. Other sources
    """
    priority_domains = {
        "github.com": 1,
        "huggingface.co": 2,
        "stackoverflow.com": 3,
        "docs.rs": 2,
        "pytorch.org": 2,
    }
    
    if relevant_urls:
        prioritized = []
        for url in relevant_urls:
            domain_score = _get_domain_priority(url, priority_domains)
            prioritized.append((url, domain_score))
        
        other_urls = []
        all_urls = set(relevant_urls)
        
        for result in search_results:
            url = result.get("url", "")
            if url and url not in all_urls:
                domain_score = _get_domain_priority(url, priority_domains)
                other_urls.append((url, domain_score + 10))
                all_urls.add(url)
        
        prioritized.extend(other_urls)
        prioritized.sort(key=lambda x: x[1])
        
        return [url for url, _ in prioritized]
    
    urls_with_scores = []
    seen_urls = set()
    
    for result in search_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            domain_score = _get_domain_priority(url, priority_domains)
            urls_with_scores.append((url, domain_score))
            seen_urls.add(url)
    
    urls_with_scores.sort(key=lambda x: x[1])
    
    return [url for url, _ in urls_with_scores]


def _get_domain_priority(url: str, priority_map: dict) -> int:
    """Get priority score for a URL based on its domain."""
    for domain, score in priority_map.items():
        if domain in url:
            return score
    return 99


def _get_source_name(url: str) -> str:
    """Extract source name from URL."""
    if "github.com" in url:
        if "/issues/" in url:
            return "GitHub Issue"
        elif "/pull/" in url:
            return "GitHub PR"
        elif "/discussions/" in url:
            return "GitHub Discussion"
        return "GitHub"
    
    if "huggingface.co" in url:
        if "/docs/" in url:
            return "HuggingFace Docs"
        return "HuggingFace"
    
    if "stackoverflow.com" in url:
        return "Stack Overflow"
    
    if "reddit.com" in url:
        return "Reddit"
    
    if "pytorch.org" in url:
        return "PyTorch Docs"
    
    return "Web"


def _extract_urls_from_results(search_results: list[dict]) -> list[str]:
    """Extract URLs from search results."""
    urls = []
    for result in search_results:
        url = result.get("url", "")
        if url and url not in urls:
            urls.append(url)
    return urls