import os
from agents import (
    WebSearchTool,
    HostedMCPTool,
    Agent,
    ModelSettings,
    RunContextWrapper,
    TResponseInputItem,
    Runner,
    RunConfig,
)
from pydantic import BaseModel
from asymetry import init_observability

init_observability()

# Tool definitions
web_search_preview = WebSearchTool(
    search_context_size="low",
    user_location={"country": "US", "type": "approximate"},
)
web_search_preview1 = WebSearchTool(
    search_context_size="low", user_location={"type": "approximate"}
)
mcp = HostedMCPTool(
    tool_config={
        "type": "mcp",
        "server_label": "gmail",
        "allowed_tools": [
            "batch_read_email",
            "get_profile",
            "get_recent_emails",
            "read_email",
            "search_email_ids",
            "search_emails",
        ],
        "authorization": os.getenv("GMAIL_OAUTH_TOKEN", ""),
        "connector_id": "connector_gmail",
        "require_approval": "always",
    }
)


class ResearchFilteringAgentSchema(BaseModel):
    research: bool


research_filtering_agent = Agent(
    name="Research Filtering Agent",
    instructions="You are a helpful assistant that can identify if a query is related to research task or a general query. If this is related to searching an email also treat it as a general query. If the query is to retrieve my email contents, treat it strictly as general query.",
    model="gpt-5",
    output_type=ResearchFilteringAgentSchema,
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)


class ResearchAgentContext:
    def __init__(self, workflow_input_as_text: str):
        self.workflow_input_as_text = workflow_input_as_text


def research_agent_instructions(
    run_context: RunContextWrapper[ResearchAgentContext],
    _agent: Agent[ResearchAgentContext],
):
    workflow_input_as_text = run_context.context.workflow_input_as_text
    return f"You are a helpful agent that can find related links from reddit only about the asked query. {workflow_input_as_text}"


research_agent = Agent(
    name="Research Agent",
    instructions=research_agent_instructions,
    model="gpt-5",
    tools=[web_search_preview],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)


agent = Agent(
    name="Agent",
    instructions="Answer this general query and use email if needed.",
    model="gpt-5",
    tools=[web_search_preview1, mcp],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)


class WorkflowInput(BaseModel):
    input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
    state = {}
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": workflow["input_as_text"]}],
        }
    ]
    research_filtering_agent_result_temp = await Runner.run(
        research_filtering_agent,
        input=[*conversation_history],
        run_config=RunConfig(
            trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_692630cf78b48190aa4803a6a794eeb70e7247e774b50f8a",
            }
        ),
    )

    conversation_history.extend(
        [item.to_input_item() for item in research_filtering_agent_result_temp.new_items]
    )

    research_filtering_agent_result = {
        "output_text": research_filtering_agent_result_temp.final_output.json(),
        "output_parsed": research_filtering_agent_result_temp.final_output.model_dump(),
    }
    if research_filtering_agent_result["output_parsed"]["research"] == True:
        research_agent_result_temp = await Runner.run(
            research_agent,
            input=[*conversation_history],
            run_config=RunConfig(
                trace_metadata={
                    "__trace_source__": "agent-builder",
                    "workflow_id": "wf_692630cf78b48190aa4803a6a794eeb70e7247e774b50f8a",
                }
            ),
            context=ResearchAgentContext(workflow_input_as_text=workflow["input_as_text"]),
        )

        conversation_history.extend(
            [item.to_input_item() for item in research_agent_result_temp.new_items]
        )

        research_agent_result = {"output_text": research_agent_result_temp.final_output_as(str)}
        return research_agent_result
    else:
        agent_result_temp = await Runner.run(
            agent,
            input=[*conversation_history],
            run_config=RunConfig(
                trace_metadata={
                    "__trace_source__": "agent-builder",
                    "workflow_id": "wf_692630cf78b48190aa4803a6a794eeb70e7247e774b50f8a",
                }
            ),
        )

        conversation_history.extend([item.to_input_item() for item in agent_result_temp.new_items])

        agent_result = {"output_text": agent_result_temp.final_output_as(str)}
        return agent_result


if __name__ == "__main__":
    import asyncio
    import time

    start = time.time()
    asyncio.run(
        run_workflow(WorkflowInput(input_as_text="What is the current stock price of Apple?"))
    )
    print(time.time() - start)
    time.sleep(6)
