import difflib
import json
import os
import re
import traceback
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.engine_loader import load_policies
from src.engine_types import PolicyError
from src.policy_engine import bin_metrics, evaluate
from src.weather import get_weather_for_location

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

book = load_policies("policies")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_retries=0)

ACTIVITY_WORDS = [
    "running", "run", "jogging", "jog", "cycling", "cycle", "biking", "bike",
    "walking", "walk", "hiking", "hike", "swimming", "swim", "trekking", "trek",
]
MODIFIER_WORDS = {
    "child": "children", "children": "children", "kid": "children", "kids": "children",
    "elderly": "elderly", "senior": "elderly", "seniors": "elderly",
    "pet": "pets", "pets": "pets", "dog": "pets", "dogs": "pets",
}
EMPTY_VALUES = {"none", "null", "", "n/a", "<city name or none>", "<activity name or none>", "<comma separated list or none>"}


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def build_alias_map(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        eid = entry["id"]
        aliases[norm(eid)] = eid
        for key in ("label", "name", "display_name", "title"):
            if isinstance(entry.get(key), str):
                aliases[norm(entry[key])] = eid
        for key in ("synonyms", "aliases", "keywords", "also"):
            vals = entry.get(key)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str):
                        aliases[norm(v)] = eid
    return aliases


def collect_activity_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for body in (book.taxonomy.get("categories") or {}).values():
        for act in body.get("activities", []):
            entries.append(act)
    return entries


def collect_modifier_entries() -> List[Dict[str, Any]]:
    return list((book.taxonomy.get("context_modifiers") or {}).get("groups", []))


ACTIVITY_ALIASES = build_alias_map(collect_activity_entries())
MODIFIER_ALIASES = build_alias_map(collect_modifier_entries())
VALID_ACTIVITIES = sorted(book.activity_category.keys())
VALID_MODIFIERS = sorted(book.modifier_ids)


def resolve(value: Optional[str], aliases: Dict[str, str]) -> Optional[str]:
    if not value:
        return None
    n = norm(value)
    if not n:
        return None
    if n in aliases:
        return aliases[n]
    if len(n) >= 3:
        for alias, target in aliases.items():
            if len(alias) >= 3 and (alias in n or n in alias):
                return target
    close = difflib.get_close_matches(n, list(aliases.keys()), n=1, cutoff=0.75)
    if close:
        return aliases[close[0]]
    return None


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


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(getattr(part, "text", "")))
        return "".join(parts)
    return str(content)


def fallback_extract(text: str):
    text_l = text.lower()
    raw_activity = next((a for a in ACTIVITY_WORDS if re.search(rf"\b{a}\b", text_l)), None)
    activity = resolve(raw_activity, ACTIVITY_ALIASES)
    match = re.search(r"\bin\s+([a-zA-Z][a-zA-Z\s]*?)(?=\s+(?:today|tomorrow|tonight|now|this|with)\b|[?.!,\x22\x27]|$)", text_l)
    city = match.group(1).strip().strip("\"'").title() if match else None
    modifiers = []
    for word, mapped in MODIFIER_WORDS.items():
        if re.search(rf"\b{word}\b", text_l):
            resolved = resolve(mapped, MODIFIER_ALIASES)
            if resolved:
                modifiers.append(resolved)
    return city, activity, sorted(set(modifiers))


def last_user_message(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return content_to_text(m.content)
    return ""


def clean_value(val: str) -> Optional[str]:
    val = val.strip().strip("*").strip()
    if val.lower() in EMPTY_VALUES:
        return None
    return val


def classify_intent_node(state: AgentState):
    history_lines = []
    for m in state["messages"][-4:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {content_to_text(m.content)}")
    history_text = "\n".join(history_lines)

    sys_prompt = (
        "Analyze the conversation and extract the target city, outdoor activity, and special modifiers. "
        "Retain any previously mentioned valid details.\n\n"
        f"Allowed activities: {', '.join(VALID_ACTIVITIES)}\n"
        f"Allowed modifiers: {', '.join(VALID_MODIFIERS)}\n"
        "ACTIVITY must be exactly one of the allowed activities, or None. "
        "MODIFIERS must only contain allowed modifiers, or None.\n\n"
        f"Conversation:\n{history_text}\n\n"
        "You MUST respond in EXACTLY this multi-line format, with no extra text:\n"
        "CITY: <city name or None>\n"
        "ACTIVITY: <activity name or None>\n"
        "MODIFIERS: <comma separated list or None>"
    )

    c_city = None
    c_act = None
    c_mods: List[str] = []

    try:
        response = llm.invoke([HumanMessage(content=sys_prompt)])
        text = content_to_text(response.content).strip()

        for line in text.split("\n"):
            line = line.strip().lstrip("-*• ").strip()
            upper = line.upper()
            if upper.startswith("CITY:"):
                c_city = clean_value(line[len("CITY:"):])
            elif upper.startswith("ACTIVITY:"):
                c_act = resolve(clean_value(line[len("ACTIVITY:"):]), ACTIVITY_ALIASES)
            elif upper.startswith("MODIFIERS:"):
                val = clean_value(line[len("MODIFIERS:"):])
                if val:
                    for item in val.split(","):
                        resolved = resolve(item.strip(), MODIFIER_ALIASES)
                        if resolved:
                            c_mods.append(resolved)
    except Exception as e:
        print("[classify_intent_node] LLM ERROR:", repr(e))
        traceback.print_exc()

    if not (c_city and c_act):
        f_city, f_act, f_mods = fallback_extract(last_user_message(state))
        c_city = c_city or f_city
        c_act = c_act or f_act
        c_mods = c_mods + f_mods

    final_city = c_city or state.get("city")
    final_act = c_act or state.get("activity")
    past_mods = state.get("modifiers") or []
    final_mods = sorted(set(past_mods + c_mods))

    print(f"[classify_intent_node] city={final_city!r} activity={final_act!r} modifiers={final_mods!r}")

    return {
        "city": final_city,
        "activity": final_act,
        "modifiers": final_mods,
        "clarification_needed": not bool(final_city and final_act),
    }


def ask_clarification_node(state: AgentState):
    city = state.get("city")
    activity = state.get("activity")
    options = ", ".join(VALID_ACTIVITIES)

    if not city and not activity:
        msg = f"I need a bit more info! Could you tell me what city you are in and what activity you are planning? I can check: {options}."
    elif not city:
        msg = f"Got it, you want to go {activity}! What city are you in?"
    else:
        msg = f"I see you are in {city}. What outdoor activity are you planning? I can check: {options}."

    return {"messages": [AIMessage(content=msg)]}


def fetch_weather_node(state: AgentState):
    try:
        weather_data = get_weather_for_location(state["city"])
    except Exception as e:
        print("[fetch_weather_node] ERROR:", repr(e))
        traceback.print_exc()
        return {"api_failed": True, "weather": None}

    if not weather_data or weather_data.get("temperature") is None:
        return {"api_failed": True, "weather": None}

    print("[fetch_weather_node] weather keys:", sorted(weather_data.keys()))
    return {"api_failed": False, "weather": weather_data}


def evaluate_policy_node(state: AgentState):
    try:
        levels = bin_metrics(book, state["weather"])
        decision = evaluate(book, state["activity"], state["modifiers"], levels)

        label = decision.decision or decision.outcome
        summary = f"**Decision:** {label}\n\n"
        if decision.fired:
            summary += f"**Primary Reason:** {decision.fired[0].advice}\n"
        elif decision.baseline_text:
            summary += f"**Note:** {decision.baseline_text}\n"
        if decision.unevaluated:
            summary += f"\n**Missing data for:** {', '.join(decision.unevaluated)}\n"

        return {
            "decision_summary": summary,
            "citations": list(decision.citations or []),
        }
    except PolicyError as e:
        print("[evaluate_policy_node] POLICY ERROR:", repr(e))
        traceback.print_exc()
        return {
            "decision_summary": f"**Decision:** Unable to evaluate safety policy.\n\n**Reason:** {e}\n",
            "citations": [],
        }
    except Exception as e:
        print("[evaluate_policy_node] ERROR:", repr(e))
        traceback.print_exc()
        return {
            "decision_summary": f"**Decision:** Unable to evaluate safety policy for these conditions.\n\n**Reason:** {type(e).__name__}: {e}\n",
            "citations": [],
        }


def handle_api_failure_node(state: AgentState):
    msg = "I'm sorry, I couldn't fetch the weather data for that location right now. Please check your spelling or try again later."
    return {"messages": [AIMessage(content=msg)]}


def generate_response_node(state: AgentState):
    sys_prompt = (
        "You are a Weather Safety Advisor. You MUST output the exact safety decision and reasoning provided below. "
        "Do NOT invent additional safety advice or ignore the provided decision.\n\n"
        f"Current Conditions:\n{json.dumps(state['weather'], indent=2, default=str)}\n\n"
        f"Deterministic Decision:\n{state['decision_summary']}"
    )
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
        text = content_to_text(response.content).strip()
    except Exception as e:
        print("[generate_response_node] LLM ERROR:", repr(e))
        traceback.print_exc()
        text = ""
    if not text:
        text = state["decision_summary"] or "I could not generate a response."
    return {"messages": [AIMessage(content=text)]}


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
    {"ask_clarification": "ask_clarification", "fetch_weather": "fetch_weather"},
)
builder.add_conditional_edges(
    "fetch_weather",
    route_after_weather,
    {"handle_api_failure": "handle_api_failure", "evaluate_policy": "evaluate_policy"},
)
builder.add_edge("evaluate_policy", "generate_response")
builder.add_edge("ask_clarification", END)
builder.add_edge("handle_api_failure", END)
builder.add_edge("generate_response", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)


def run_weather_agent(user_input: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        answer_text = content_to_text(result["messages"][-1].content)

        return {
            "answer": answer_text,
            "citations": result.get("citations", []),
        }

    except Exception as e:
        print("[run_weather_agent] ERROR:", repr(e))
        traceback.print_exc()
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "answer": "⏳ **API Rate Limit Reached:** Google's free-tier limit temporarily exceeded. Please wait 30 seconds and retry.",
                "citations": [],
            }
        elif "503" in error_msg or "UNAVAILABLE" in error_msg:
            return {
                "answer": "🌩️ **Google API Overloaded:** Google's servers are currently experiencing high global demand. Please try your request again in a minute!",
                "citations": [],
            }
        elif "404" in error_msg or "NOT_FOUND" in error_msg:
            return {
                "answer": f"⚠️ **Model Not Found:** The model '{MODEL_NAME}' is not available for your API key. Set GEMINI_MODEL in your .env to a valid model name.",
                "citations": [],
            }
        else:
            return {
                "answer": f"⚠️ **System Error:** {error_msg[:100]}...\n\nPlease try again.",
                "citations": [],
            }