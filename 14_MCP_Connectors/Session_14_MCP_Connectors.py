import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session 14: MCP-Powered X Post Summarizer

    ## Building a LangGraph Agent with GitHub MCP Tools, X API Tools, and Memory

    ## Learning Objectives:

    - **Ingest MCP servers as LangGraph tools** using `langchain-mcp-adapters` to connect to the GitHub MCP Server and use its tools programmatically
    - **Wrap the X (Twitter) API as a LangChain tool** using the `@tool` decorator so a LangGraph agent can search and retrieve public posts
    - **Build a LangGraph agent with memory** that combines MCP-sourced tools and custom tools, using `MemorySaver` for short-term conversational memory
    - **Orchestrate a full workflow through the agent** — search X posts, generate summaries, create a GitHub repo, commit files, branch, and open a PR — all via natural language

    ## Overview

    In this notebook, you will build a **LangGraph ReAct agent** that has access to two categories of tools:

    1. **GitHub MCP tools** — loaded from the official GitHub MCP Server via `langchain-mcp-adapters`. These replace manual `git` commands with tool calls the agent can invoke (create repos, commit files, create branches, open PRs).
    2. **X API tools** — custom Python functions wrapped with the `@tool` decorator that call the X API v2 directly to search and retrieve posts.

    The agent uses **`MemorySaver`** for short-term memory so it can maintain context across multi-step workflows within a conversation thread.

    There will be one breakout room with two phases:

    - 🤝 Phase 1: Setup, Tools & Agent Construction
      - Task 1: Dependencies & Environment
      - Task 2: X API as LangChain Tools
      - Task 3: Connect to GitHub MCP Server & Load Tools
      - Task 4: Build the LangGraph Agent with Memory
      - Task 5: Test the Agent — Search & Summarize X Posts
      - Activity #1: Extend the Agent with a Custom X API Tool
    - 🤝 Phase 2: MCP Workflow Through the Agent
      - Task 6: Create a GitHub Repository
      - Task 7: Commit the Summary
      - Task 8: Create a Feature Branch & Add Metadata
      - Task 9: Open a Pull Request
      - Task 10: Commit the X API Script
      - Task 11: Update the README
      - Activity #1: Multi-Account Comparison Pipeline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 🤝 Breakout Room
    ## Setup, Tools & Agent Construction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1: Dependencies & Environment

    We need:
    - `langchain-mcp-adapters` to connect to MCP servers and convert their tools into LangChain tools
    - `langgraph` for our agent graph with memory
    - `langchain-openai` for our LLM
    - `requests` for the X API calls
    - `nest-asyncio` for async MCP operations inside Jupyter

    > NOTE: Create a `.env` file in this directory with `X_BEARER_TOKEN`, `OPENAI_API_KEY`, and `GITHUB_PAT` before running.
    >
    > Setup references:
    > - GitHub fine-grained PAT guide: [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token)
    > - X API Bearer Token setup: [X Developer Portal](https://developer.x.com/en/portal/dashboard) and [X API access tiers](https://developer.x.com/en/products/twitter-api)
    """)
    return


@app.cell
def _():
    import os
    import getpass
    from dotenv import load_dotenv

    load_dotenv()

    if not os.environ.get("X_BEARER_TOKEN"):
        os.environ["X_BEARER_TOKEN"] = getpass.getpass(
            "Enter your X Bearer Token:"
        )

    if not os.environ.get("GITHUB_PAT"):
        os.environ["GITHUB_PAT"] = getpass.getpass("Enter your GitHub PAT:")

    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    return (os,)


@app.cell
def _():
    import nest_asyncio

    nest_asyncio.apply()  # Required for async operations in Jupyter
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Getting Your Credentials

    **GitHub PAT (fine-grained):**
    1. Open [GitHub Personal Access Tokens (fine-grained)](https://github.com/settings/personal-access-tokens/new).
    2. Follow [GitHub's PAT setup guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).
    3. Set repository permissions to at least:
       - `Contents`: Read and write
       - `Pull requests`: Read and write
       - `Metadata`: Read-only

    **X Bearer Token:**
    1. Open the [X Developer Portal](https://developer.x.com/en/portal/dashboard).
    2. Create/select a Project + App, then go to **Keys and Tokens** to generate a Bearer Token.
    3. Confirm your plan supports the recent search endpoint (`GET /2/tweets/search/recent`) from the [X API product page](https://developer.x.com/en/products/twitter-api).
    """)
    return


@app.cell
def _():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        base_url="http://192.168.1.185:8080/v1",
    )

    # Test the connection
    response = llm.invoke("Say 'MCP agent ready!' in exactly those words.")
    print(response.content)
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2: X API as LangChain Tools

    Instead of relying on a community-built MCP server for X, we'll call the **X API v2** directly and wrap our functions with the `@tool` decorator. This makes them available to our LangGraph agent as callable tools — just like the MCP tools will be.

    This is a key architectural decision: **not everything needs to be an MCP server**. Wrapping a simple API call as a `@tool` is often simpler and more transparent.

    **📚 Documentation:**
    - [LangChain Tools Conceptual Guide](https://python.langchain.com/docs/concepts/tools/)
    - [X API v2 Documentation](https://developer.x.com/en/docs/x-api)
    """)
    return


@app.cell
def _(os):
    import requests
    import json
    from langchain_core.tools import tool

    BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN")


    @tool
    def search_recent_posts(query: str, max_results: int = 5) -> str:
        """Search recent X/Twitter posts using the v2 API.
        Returns posts from the last 7 days matching the query.
        Use this for keyword searches, hashtag searches, or general topic searches.

        Args:
            query: The search query (e.g., 'AI safety', '#machinelearning', 'from:AndrewYNg')
            max_results: Number of results to return (10-100, default 20)
        """
        url = "https://api.x.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        params = {
            "query": query,
            "max_results": min(max(max_results, 10), 100),
            "tweet.fields": "created_at,public_metrics,author_id,text",
            "expansions": "author_id",
            "user.fields": "name,username",
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        tweets = data.get("data", [])
        if not tweets:
            return "No posts found for this query."

        result_lines = [f"Found {len(tweets)} posts:\n"]
        for t in tweets:
            metrics = t.get("public_metrics", {})
            result_lines.append(
                f"[{t.get('created_at', 'unknown')[:10]}] "
                f"{t['text'][:200]}\n"
                f"  Likes: {metrics.get('like_count', 0)} | "
                f"Retweets: {metrics.get('retweet_count', 0)}"
            )
        return "\n\n".join(result_lines)


    @tool
    def get_user_posts(username: str, max_results: int = 20) -> str:
        """Get recent original posts (no retweets) from a specific X/Twitter user.
        Use this when you want to see what a specific account has been posting.

        Args:
            username: The X/Twitter handle without the @ sign (e.g., 'AndrewYNg')
            max_results: Number of results to return (10-100, default 20)
        """
        query = f"from:{username} -is:retweet"
        return search_recent_posts.invoke(
            {"query": query, "max_results": max_results}
        )


    x_api_tools = [search_recent_posts, get_user_posts]
    print(
        f"Created {len(x_api_tools)} X API tools: {[t.name for t in x_api_tools]}"
    )
    return BEARER_TOKEN, get_user_posts, requests, tool, x_api_tools


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's verify our X API tools work before wiring them into the agent:
    """)
    return


@app.cell
def _(get_user_posts):
    # Quick test — fetch recent posts from a public account
    _result = get_user_posts.invoke({"username": "llm_wizard", "max_results": 10})
    print(_result[:500])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 3: Connect to GitHub MCP Server & Load Tools

    Now we'll connect to the **GitHub MCP Server** — an official, GitHub-maintained MCP server that gives agents the ability to manage repositories, issues, pull requests, and more.

    We use `langchain-mcp-adapters` to:
    1. Connect to the remote GitHub MCP server over HTTP
    2. Automatically convert all MCP tools into LangChain-compatible tools

    This is the key MCP integration point — instead of writing custom GitHub API wrappers, we get a full set of tools for free just by connecting to the MCP server.

    **📚 Documentation:**
    - [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
    - [GitHub MCP Server](https://github.com/github/github-mcp-server)
    - [Model Context Protocol Specification](https://modelcontextprotocol.io/)
    """)
    return


@app.cell
async def _(os):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # Connect to the GitHub MCP server using Streamable HTTP transport
    # The server exposes GitHub operations as MCP tools that our agent can call
    mcp_client = MultiServerMCPClient(
        {
            "github": {
                "transport": "http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": f"Bearer {os.environ['GITHUB_PAT']}",
                },
            }
        }
    )

    # Load all tools from the MCP server
    github_mcp_tools = await mcp_client.get_tools()

    print(f"Loaded {len(github_mcp_tools)} GitHub MCP tools:\n")
    for t in github_mcp_tools:
        print(f"  - {t.name}: {t.description[:80]}...")
    return (github_mcp_tools,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key GitHub MCP Tools

    The MCP server exposes many tools, but the key ones we'll use are:

    | MCP Tool | Replaces (Git CLI) | What It Does |
    |---|---|---|
    | `create_repository` | `git init` + GitHub UI | Creates a new repo on your account |
    | `create_or_update_file` | `git add` + `git commit` + `git push` | Commits a file directly to a branch |
    | `create_branch` | `git checkout -b` | Creates a new branch |
    | `create_pull_request` | `gh pr create` | Opens a PR from one branch to another |
    | `search_repositories` | `gh repo list` | Searches across your repos |
    | `get_file_contents` | `git show` / `cat` | Reads a file from a repo |
    | `list_commits` | `git log` | Shows commit history |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 4: Build the LangGraph Agent with Memory

    Now we combine **both tool sets** into a single LangGraph agent:
    - **X API tools** — custom `@tool` functions for searching posts
    - **GitHub MCP tools** — loaded from the MCP server via `langchain-mcp-adapters`

    We add **`MemorySaver`** for short-term memory so the agent remembers context across the multi-step workflow (e.g., it fetches posts in one turn, summarizes them in the next, and commits the summary in a third).

    The architecture follows the standard LangGraph ReAct pattern from Sessions 4-6:

    ```
    ┌─────────┐     ┌───────────┐
    │  START   │────▶│   Agent   │◀──────────────┐
    └─────────┘     │  (LLM +   │               │
                    │   tools)  │               │
                    └─────┬─────┘               │
                          │                     │
                   has tool calls?              │
                    /           \               │
                  yes            no             │
                  /               \             │
        ┌─────────────┐     ┌─────────┐        │
        │  Tool Node  │     │   END   │        │
        │ (X API +    │     └─────────┘        │
        │  GitHub MCP)│─────────────────────────┘
        └─────────────┘
    ```

    **📚 Documentation:**
    - [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
    - [MemorySaver (Checkpointing)](https://langchain-ai.github.io/langgraph/concepts/persistence/)
    - [ToolNode Prebuilt](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)
    """)
    return


@app.cell
def _(github_mcp_tools, x_api_tools):
    from typing import Annotated, Literal
    from typing_extensions import TypedDict

    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    # Combine all tools: X API tools + GitHub MCP tools
    all_tools = x_api_tools + github_mcp_tools
    print(f"Total tools available to agent: {len(all_tools)}")
    print(f"  X API tools: {[t.name for t in x_api_tools]}")
    print(f"  GitHub MCP tools: {[t.name for t in github_mcp_tools]}")
    return (
        Annotated,
        END,
        HumanMessage,
        Literal,
        MemorySaver,
        START,
        StateGraph,
        SystemMessage,
        ToolNode,
        TypedDict,
        add_messages,
        all_tools,
    )


@app.cell
def _(
    Annotated,
    END,
    Literal,
    MemorySaver,
    START,
    StateGraph,
    SystemMessage,
    ToolNode,
    TypedDict,
    add_messages,
    all_tools,
    llm,
):
    # Step 1: Define the Agent State
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]


    # Step 2: Define the system prompt
    SYSTEM_PROMPT = """You are an AI assistant that can search X/Twitter posts and manage GitHub repositories.

    You have two categories of tools:
    1. X API tools: search_recent_posts, get_user_posts — for searching and retrieving X/Twitter posts
    2. GitHub MCP tools: for creating repos, committing files, creating branches, opening PRs, etc.

    When asked to summarize posts, retrieve them first using the X API tools, then provide a structured
    markdown summary with: Overview, Key Themes, Notable Posts, and Summary Statistics.

    When asked to perform GitHub operations, use the appropriate GitHub MCP tool.
    Always use the available tools when appropriate. Be concise in your responses."""

    # Step 3: Bind tools to the LLM
    llm_with_tools = llm.bind_tools(all_tools)


    # Step 4: Define the agent node
    def agent_node(state: AgentState):
        """The agent node — calls the LLM with the current conversation and available tools."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}


    # Step 5: Define the tool node
    tool_node = ToolNode(all_tools, handle_tool_errors=True)


    # Step 6: Define routing logic
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine whether to call tools or end the conversation."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"


    # Step 7: Build the graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    # Compile with MemorySaver for short-term memory across turns
    checkpointer = MemorySaver()
    agent = workflow.compile(checkpointer=checkpointer)

    print("Agent compiled with memory and tools!")
    return (agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's visualize the graph to confirm our architecture:
    """)
    return


@app.cell
def _(agent):
    from IPython.display import Image, display

    display(Image(agent.get_graph().draw_mermaid_png()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Helper function for running the agent

    We'll use a single `thread_id` throughout the notebook so the agent remembers previous interactions (short-term memory via the checkpointer).
    """)
    return


@app.cell
def _(HumanMessage, agent):
    # Use a consistent thread_id so the agent remembers context across all tasks
    config = {"configurable": {"thread_id": "mcp-workflow-1"}}


    async def ask_agent(user_message: str) -> str:
        """Send a message to the agent and return its final response."""
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config,
        )
        return response["messages"][-1].content

    return (ask_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 5: Test the Agent — Search & Summarize X Posts

    Let's put the agent through its paces. First, we'll ask it to search for posts and generate a summary. Because we're using `MemorySaver` with a consistent `thread_id`, the agent will remember the posts it found when we ask it to summarize them.
    """)
    return


@app.cell
async def _(ask_agent):
    # Ask the agent to fetch posts — it will use the get_user_posts tool
    _result = await ask_agent("Get recent posts from @llm_wizard on X/Twitter.")
    print(_result[:1000])
    return


@app.cell
async def _(ask_agent):
    # Ask the agent to summarize — it remembers the posts from the previous turn!
    summary = await ask_agent(
        "Now summarize those posts into a structured markdown report with sections for: "
        "Overview, Key Themes, Notable Posts, and Summary Statistics. "
        "Format it so it can be saved directly as a summary.md file."
    )
    print(summary)
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save the summary locally for reference:
    """)
    return


@app.cell
def _(summary):
    with open("summary.md", "w") as f:
        f.write(summary)

    print("Summary saved to summary.md")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❓ Question #1:



    ##### Answer:

    *I don't know what the question is, but I assume it's whether summary.md is created, which it is.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🏗️ Activity #1:

    Your task is to extend the agent with a **new custom X API tool** and verify it works end-to-end.

    1. **Create a new `@tool` function** called `get_user_profile` that retrieves a user's public profile information using the X API v2 [`GET /2/users/by/username/:username`](https://developer.x.com/en/docs/x-api/users/lookup/api-reference/get-users-by-username-username) endpoint. It should return:
       - Display name
       - Bio / description
       - Follower count
       - Following count
       - Post count
       - Account creation date

    2. **Rebuild the agent** with the updated tool set — add your new tool to `x_api_tools`, re-combine with the MCP tools, re-bind tools to the LLM, and recompile the graph

    3. **Test it** by asking the agent to:
       - Retrieve the profile of an AI thought leader of your choice
       - Compare that profile with the posts you already retrieved in Task 5 — does the bio match the posting themes?

    > HINT: The X API v2 user lookup endpoint uses the same Bearer Token authentication. You'll need `user.fields=description,public_metrics,created_at` in your request params.
    """)
    return


@app.cell
async def _(BEARER_TOKEN, ask_agent, requests, tool):
    @tool
    def get_user_profile(username: str) -> str:
        """Retrieve a user's public profile information from X/Twitter.

        Returns the display name, bio/description, follower count, following count,
        post count, and account creation date.

        Args:
            username: The X/Twitter handle without the @ sign (e.g., 'AndrewYNg')
        """
        url = f"https://api.x.com/2/users/by/username/{username}"
        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        params = {"user.fields": "description,public_metrics,created_at"}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            return f"User @{username} not found."

        user = data["data"]
        metrics = user.get("public_metrics", {})

        profile_info = [
            f"Display Name: {user.get('name', 'N/A')}",
            f"Username: @{user.get('username', 'N/A')}",
            f"Bio/Description: {user.get('description', 'N/A')}",
            f"Followers: {metrics.get('followers_count', 0):,}",
            f"Following: {metrics.get('following_count', 0):,}",
            f"Posts: {metrics.get('tweet_count', 0):,}",
            f"Account Created: {user.get('created_at', 'N/A')}",
        ]

        return "\n".join(profile_info)

    print(f"Created new tool: get_user_profile")

    x_api_tools_updated = [get_user_profile]
    print(f"Updated X API tools: {[t.name for t in x_api_tools_updated]}")

    profile_result = await ask_agent(
            "Use the get_user_profile tool to retrieve the profile information for AndrewYNg on X/Twitter."
        )
    print("=== Profile Result ===")
    print(profile_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Phase 2: MCP Workflow Through the Agent

    Now we'll use the same agent to perform all GitHub repository operations through the **GitHub MCP tools**. Because the agent has memory, it already knows the summary it generated in Phase 1.

    Each task below sends a natural language instruction to the agent. The agent decides which GitHub MCP tool(s) to call to fulfill the request.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 6: Create a New Repository

    The agent will use the `create_repository` MCP tool to create a new repo on your GitHub account.
    """)
    return


@app.cell
async def _(ask_agent):
    _result = await ask_agent(
        "Using your GitHub tools, create a new public repository on my account called `x-post-summarizer-2026`. Add a description: 'AI-generated summary of a public figure's 2026 X posts, built with LangGraph, MCP tools, and the X API.' Initialize it with a README."
    )
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 7: Commit the Summary to Your Repo

    The agent remembers the summary it generated earlier (short-term memory) and can commit it directly.
    """)
    return


@app.cell
async def _(ask_agent):
    _result = await ask_agent(
        "Using your GitHub tools, create a new file called `summary.md` in the `x-post-summarizer-2026` repo on the `main` branch. The file should contain the X post summary you generated earlier. Use the commit message: 'Add 2026 X post summary'."
    )
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 8: Create a Feature Branch and Add Metadata

    The agent will use `create_branch` and `create_or_update_file` MCP tools.
    """)
    return


@app.cell
async def _(ask_agent):
    _result = await ask_agent(
        "Create a new branch called `add-metadata` in my `x-post-summarizer-2026` repo. On that branch, create a file called `metadata.json` that contains: the account handle analyzed, the date range of posts, the number of posts analyzed, and the top 5 themes identified from the summary. Commit it with the message 'Add analysis metadata'."
    )
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 9: Open a Pull Request

    The agent will use the `create_pull_request` MCP tool.
    """)
    return


@app.cell
async def _(ask_agent):
    _result = await ask_agent(
        "Open a pull request in my `x-post-summarizer-2026` repo from the `add-metadata` branch to `main`. Title it 'Add analysis metadata' and include a description summarizing what the metadata file contains."
    )
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 10: Commit the X API Script

    We'll ask the agent to commit a clean version of the X search script — reading credentials from environment variables, no hardcoded keys.
    """)
    return


@app.cell
async def _(ask_agent):
    x_search_script = 'import requests\nimport json\nimport os\nfrom datetime import datetime\n\nBEARER_TOKEN = os.environ.get("X_BEARER_TOKEN")\n\ndef search_recent_posts(query: str, max_results: int = 20) -> dict:\n    """Search recent X posts using the v2 API."""\n    url = "https://api.x.com/2/tweets/search/recent"\n    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}\n    params = {\n        "query": query,\n        "max_results": min(max_results, 100),\n        "tweet.fields": "created_at,public_metrics,author_id,text",\n        "expansions": "author_id",\n        "user.fields": "name,username",\n    }\n    response = requests.get(url, headers=headers, params=params)\n    response.raise_for_status()\n    return response.json()\n\ndef get_user_posts(username: str, max_results: int = 20) -> dict:\n    """Get recent posts from a specific user."""\n    query = f"from:{username} -is:retweet"\n    return search_recent_posts(query, max_results)\n\nif __name__ == "__main__":\n    import sys\n    handle = sys.argv[1] if len(sys.argv) > 1 else "llM_wizard"\n    print(f"Searching for recent posts from @{handle}...")\n    results = get_user_posts(handle)\n    with open("posts.json", "w") as f:\n        json.dump(results, f, indent=2)\n    tweets = results.get("data", [])\n    print(f"Found {len(tweets)} posts.")\n    for tweet in tweets:\n        print(f"  [{tweet["created_at"][:10]}] {tweet["text"][:100]}...")\n'
    _result = await ask_agent(
        f"Using your GitHub tools, create a new file called `x_search.py` in the `x-post-summarizer-2026` repo on the `main` branch. Use the commit message: 'Add X API search script'. Here is the file content:\n\n{x_search_script}"
    )
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 11: Update the README

    The agent already knows everything about the project from its conversation memory — what account was analyzed, how the project was built, etc.
    """)
    return


@app.cell
async def _(ask_agent):
    _result = await ask_agent(
        "Update the README.md in my `x-post-summarizer-2026` repo on main to include: a project description explaining this repo summarizes a public figure's 2026 X posts using AI, the handle analyzed, how the project was built (using a LangGraph agent with GitHub MCP tools for repo operations and the X API v2 for post retrieval), and instructions for someone else to replicate the process — including how to set up their X API Bearer Token and install Python dependencies."
    )
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❓ Question #2:

    Compare using GitHub MCP tools (through a LangGraph agent) to traditional `git` commands. What felt easier? What felt harder or less transparent?

    ##### Answer:

    GitHub MCP tools through the agent made repository operations feel smoother since I could just describe what I wanted in plain language rather than looking up git command syntax. But it felt less transparent because I couldn't easily see exactly what underlying operations were happening — there's no equivalent to running `git status` or `git log` to verify what's going on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❓ Question #3:

    You used MCP for GitHub but wrapped the X API as a `@tool` directly. What are the tradeoffs of consuming an API through an MCP server versus wrapping it as a LangChain tool? When would each approach make more sense?

    ##### Answer:

    MCP servers give you standardized, automatically-discoverable tools with zero implementation overhead — but you're dependent on external server availability and have less control over the underlying logic. Direct `@tool` wrappers give you full transparency, immediate debugging, and complete control, but require more manual code and aren't portable across different agent frameworks. Use MCP when a well-maintained server already exists for your needs (like GitHub), and use `@tool` wrappers when you need quick custom integrations, have simple API needs, or want precise control without the MCP infrastructure layer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🏗️ Activity #1:

    Your task is to extend the MCP workflow by building a **Multi-Account Comparison Pipeline** through the agent.

    You are expected to:

    1. **Retrieve posts from a second X account** — choose another public figure or thought leader in a related field

    2. **Generate a structured comparison** by asking the agent to create a `comparison.md` file that includes:
       - Side-by-side topic analysis for both accounts
       - Tone and sentiment differences
       - Posting frequency comparison
       - Top 3 most notable posts from each account
       - A brief conclusion about each account's focus area

    3. **Commit through the MCP workflow**:
       - Create a new branch called `add-comparison` in your `x-post-summarizer-2026` repo
       - Commit `comparison.md` to that branch
       - Open a pull request to merge it into `main`

    > NOTE: The agent already has memory of the first account's posts from Phase 1. You only need to fetch posts from the second account — the agent will use its memory for the rest.
    """)
    return


@app.cell
async def _(ask_agent):
    second_account = "karpathy"

    posts_result = await ask_agent(
        f"Get recent posts from @{second_account} on X/Twitter so I can compare them with the llm_wizard posts we fetched earlier."
    )
    print("=== First: Posts from second account ===")
    print(posts_result[:800])

    comparison_content = await ask_agent(
        f"Based on all the posts you've seen from both @{second_account} and llm_wizard, "
        "create a detailed comparison.md file with the following sections:\n"
        "1. **Account Overview** - Brief description of each account's focus\n"
        "2. **Topic Analysis** - Side-by-side comparison of main topics discussed by each\n"
        "3. **Tone and Sentiment** - Differences in writing style, sentiment, and approach\n"
        "4. **Posting Frequency** - Comparison of when and how often each posts\n"
        "5. **Top 3 Notable Posts** - From each account, including the post content and why it's notable\n"
        "6. **Focus Area Conclusion** - Brief conclusion about each account's core focus and expertise area\n\n"
        "Format everything in clean markdown with clear headings. Make it comprehensive and insightful."
    )

    comparison_result = await ask_agent(
        f"Using your GitHub tools, create a new branch called 'add-comparison' in my x-post-summarizer-2026 repo. "
        "Then create a file called comparison.md on that branch with the following content:\n\n{comparison_content}\n\n"
        "Commit it with the message: 'Add multi-account comparison between llm_wizard and "
        + second_account
        + "'"
    )
    print(comparison_result)

    pr_result = await ask_agent(
        "Open a pull request in my x-post-summarizer-2026 repo from the add-comparison branch to main. "
        "Title it 'Add multi-account comparison' and include a description explaining this adds a detailed comparison "
        f"between llm_wizard and @{second_account} covering topics, tone, and notable posts."
    )
    print(pr_result)
    return


if __name__ == "__main__":
    app.run()
