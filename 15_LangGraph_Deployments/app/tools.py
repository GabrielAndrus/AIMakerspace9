"""Toolbelt assembly for agents.

Collects third-party tools and local tools (like RAG) into a single list that
graphs can bind to their language models.
"""

from __future__ import annotations

from typing import List

from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from app.rag import retrieve_information


@tool
def searx_search(query: str) -> str:
    """Search the web using SearXNG."""
    searx = SearxSearchWrapper(searx_host="http://192.168.1.36:4000")
    return searx.run(query)


def get_tool_belt() -> List:
    """Return the list of tools available to agents (SearXNG, Arxiv, RAG)."""
    return [searx_search, ArxivQueryRun(), retrieve_information]
