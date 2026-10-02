import json
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.engine_loader import load_policies
from src.policy_engine import bin_metrics, evaluate
from src.weather import get_weather_for_location

load_dotenv()

book = load_policies("policies")
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=0)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    city: Optional[str]
    activity: Optional[str]
    modifiers: List[str]
    clarification_needed: bool
    api_failed: bool
    weather: Optional[Dict[str, Any]]
    decision_summary: Optional[str]
    citations: List[str]

def classify_intent_node(state: AgentState):
    """Extracts intent and merges with existing state to prevent data loss across turns."""
    sys_prompt = """You are an intent extraction router.
    Analyze the conversation. Extract the intended city, outdoor activity, and any special modifiers (e.g., 'elderly', 'child').
    Return ONLY valid JSON matching this schema:
    {
      "city": "string or null",
      "activity": "string or null",
      "modifiers": ["list of strings"]
    }"""
    
    response = llm.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        current_city = data.get("city") or state.get("city")
        current_activity = data.get("activity") or state.get("activity")
        
        past_mods = state.get("modifiers") or []
        new_mods = data.get("modifiers") or []
        combined_mods = list(set(past_mods + new_mods))
        
        needs_clarification = not bool(current_city and current_activity)
        
        return {
            "city": current_city,
            "activity": current_activity,
            "modifiers": combined_mods,
            "clarification_needed": needs_clarification
        }
    except Exception:
        needs_clarification = not bool(state.get("city") and state.get("activity"))
        return {"clarification_needed": needs_clarification}

def ask_clarification_node(state: AgentState):
    """Dynamically prompts the user for missing required fields."""
    city = state.get("city")
    activity = state.get("activity")
    
    if not city and not activity:
        msg = "I need a bit more info! Could you tell me what city you are in and what activity you are planning?"
    elif not city:
        msg = f"Got it, you want to go {activity}! What city are you in?"
    else:
        msg = f"I see you are in {city}. What outdoor activity are you planning?"
        
    return {"messages": [AIMessage(content=msg)]}

def fetch_weather_node(state: AgentState):
    """Fetches weather data via external API."""
    weather_data = get_weather_for_location(state["city"])
    
    if not weather_data or weather_data.get("temperature") is None:
        return {"api_failed": True, "weather": None}
        
    return {"api_failed": False, "weather": weather_data}

def evaluate_policy_node(state: AgentState):
    """Executes the deterministic policy engine against active weather conditions."""
    levels = bin_metrics(book, state["weather"])
    decision = evaluate(book, state["activity"], state["modifiers"], levels)
    
    summary = f"**Decision:** {decision.decision or decision.outcome}\n\n"
    if decision.fired:
        summary += f"**Primary Reason:** {decision.fired[0].advice}\n"
    
    return {
        "decision_summary": summary,
        "citations": decision.citations
    }

def handle_api_failure_node(state: AgentState):
    """Provides a graceful fallback when the weather API is unreachable."""
    msg = "I'm sorry, I couldn't fetch the weather data for that location right now. Please check your spelling or try again later."
    return {"messages": [AIMessage(content=msg)]}

def generate_response_node(state: AgentState):
    """Translates the deterministic engine output into natural language."""
    sys_prompt = f"""You are a Weather Safety Advisor. 
    You MUST output the exact safety decision and reasoning provided below. 
    Do NOT invent additional safety advice or ignore the provided decision.
    
    Current Conditions:
    {json.dumps(state['weather'], indent=2)}
    
    Deterministic Decision:
    {state['decision_summary']}
    """
    
    response = llm.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    return {"messages": [response]}

def route_after_intent(state: AgentState):
    if state.get("clarification_needed"):
        return "ask_clarification"
    return "fetch_weather"

def route_after_weather(state: AgentState):
    if state.get("api_failed"):
        return "handle_api_failure"
    return "evaluate_policy"


builder = StateGraph(AgentState)

builder.add_node("classify_intent", classify_intent_node)
builder.add_node("ask_clarification", ask_clarification_node)
builder.add_node("fetch_weather", fetch_weather_node)
builder.add_node("evaluate_policy", evaluate_policy_node)
builder.add_node("handle_api_failure", handle_api_failure_node)
builder.add_node("generate_response", generate_response_node)

builder.add_edge(START, "classify_intent")
builder.add_conditional_edges(
    "classify_intent", 
    route_after_intent,
    {"ask_clarification": "ask_clarification", "fetch_weather": "fetch_weather"}
)
builder.add_conditional_edges(
    "fetch_weather",
    route_after_weather,
    {"handle_api_failure": "handle_api_failure", "evaluate_policy": "evaluate_policy"}
)
builder.add_edge("evaluate_policy", "generate_response")
builder.add_edge("ask_clarification", END)
builder.add_edge("handle_api_failure", END)
builder.add_edge("generate_response", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

def run_weather_agent(user_input: str, session_id: str):
    """Entry point for the Streamlit UI."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        
        raw_content = result["messages"][-1].content
        if isinstance(raw_content, list):
            answer_text = raw_content[0].get("text", "") if len(raw_content) > 0 else str(raw_content)
        else:
            answer_text = str(raw_content)
            
        return {
            "answer": answer_text,
            "citations": result.get("citations", [])
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "answer": "⏳ **API Rate Limit Reached:** Google's free-tier limit temporarily exceeded. Please wait 30 seconds and retry.",
                "citations": []
            }
        elif "503" in error_msg or "UNAVAILABLE" in error_msg:
            return {
                "answer": "🌩️ **Google API Overloaded:** Google's servers are currently experiencing high global demand. Please try your request again in a minute!",
                "citations": []
            }
        else:
            return {
                "answer": f"⚠️ **System Error:** {error_msg[:100]}...\n\nPlease try again.",
                "citations": []
            }