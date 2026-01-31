import operator
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# --- 1. Define the Shared State ---
class AgentState(TypedDict):
    # 'messages' stores the conversation history
    # 'next_step' helps the graph decide where to go
    messages: Annotated[List[str], operator.add]
    files_found: List[str]
    report_ready: bool


# --- 2. Define the Nodes (The Agents) ---
llm = ChatOpenAI(model="gpt-4o")


def analyst_node(state: AgentState):
    """Simulates the File Analyst reading local context."""
    # In a real LangGraph setup, you'd bind a FileSystem tool here
    print("--- 🔍 Analyst is scanning files ---")
    return {
        "messages": ["Analyst: Found logs.txt and notes.txt. Created summary_report.md."],
        "files_found": ["logs.txt", "notes.txt"],
        "report_ready": True,
    }


def manager_node(state: AgentState):
    """Checks if the report is sufficient."""
    print("--- 👔 Manager is reviewing work ---")
    if len(state["files_found"]) > 0:
        return {"messages": ["Manager: Great job. Work approved."]}
    return {"messages": ["Manager: No files found. Try again."]}


# --- 3. Build the Graph Architecture ---
workflow = StateGraph(AgentState)

# Add our agents as nodes
workflow.add_node("analyst", analyst_node)
workflow.add_node("manager", manager_node)

# Define the edges (The Logic Flow)
workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "manager")
workflow.add_edge("manager", END)

# Compile the graph
app = workflow.compile()


# --- 4. Execute ---
async def main():
    inputs = {"messages": ["User: Summarize my folder."], "files_found": [], "report_ready": False}
    async for output in app.astream(inputs):
        for key, value in output.items():
            print(f"Node '{key}': {value['messages'][-1]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
