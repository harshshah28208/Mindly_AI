"""LangGraph StateGraph implementation for Mindly Lab 3.

Architectural Flow (Agentic Lifecycle):
UNDERSTAND -> ANALYZE -> DECIDE -> PLAN -> SELECT TOOL -> EXECUTE -> EVALUATE -> RESPOND

Features:
- Deterministic Safety Pre-Check (HIGH risk directly routes to crisis support & supervisor alert)
- Human Supervisor Escalation in demo mode (non-emergency project owner alert with data minimization)
- Multimodal Input Normalization (Text, Voice transcript, Image vision analysis)
- Selective Gale Encyclopedia of Medicine RAG pipeline
- Persistent SQLite Memory Vault, Wellness Journal, and Goal & Habit Planner
- Dynamic Tool Execution (Grounding, Coping, RAG, Vision, Journal, Goals, Memory, Direct Response)
- Evaluator Node with bounded retry/replanning loop (MAX_RETRIES = 2)
- StateGraph with MemorySaver checkpointer for thread-level multi-turn memory
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.emotion import classify_emotion
from agent.evaluator import evaluate_execution
from agent.intent import classify_intent, is_physical_health_query
from agent.planner import plan_action
from agent.safety import (
    assess_safety,
    generate_crisis_response,
    generate_third_person_support_response,
    normalize_input,
    screen_urgent_medical,
)
from agent.state import AgentState
from config.settings import settings
from tools.coping import generate_coping_strategies
from tools.direct_response import generate_direct_response
from tools.escalation import escalation_service
from tools.goals import handle_goal_command
from tools.grounding import interactive_grounding_tool
from tools.journal import handle_journal_command
from tools.memory import handle_memory_command
from tools.rag import execute_rag_query
from tools.vision import analyze_image


def get_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOllama:
    """Instantiate and return configured ChatOllama client."""
    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=base_url or settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.ollama_temperature,
    )


# ------------------------------------------------------------------------------
# NODE DEFINITIONS
# ------------------------------------------------------------------------------
def create_safety_guard_node():
    """Create safety node executing deterministic risk assessment and input normalization."""
    def safety_guard_node(state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        if not user_input and state.get("messages"):
            last_msg = state["messages"][-1]
            user_input = str(last_msg.content)

        normalized = normalize_input(user_input)
        safety_eval = assess_safety(normalized)
        urgent_med = screen_urgent_medical(normalized)

        health_prio = "NONE"
        if urgent_med["is_urgent"]:
            health_prio = "URGENT"
        elif is_physical_health_query(normalized):
            health_prio = "ROUTINE"

        return {
            "user_input": user_input,
            "normalized_input": normalized,
            "risk_level": str(safety_eval["risk_level"]),
            "safety_status": str(safety_eval["safety_status"]),
            "health_priority": health_prio,
            "retry_count": state.get("retry_count", 0),
            "replan_count": state.get("replan_count", 0),
        }
    return safety_guard_node


def create_crisis_node():
    """Create crisis node with hotline resources and supervisor escalation."""
    def crisis_node(state: AgentState) -> Dict[str, Any]:
        crisis_text = generate_crisis_response()
        user_text = state.get("user_input", "")
        
        # Trigger supervisor escalation in demo mode (data minimization)
        escalation_result = escalation_service.escalate(
            risk_level="HIGH",
            input_source=state.get("input_type", "text"),
            safety_summary="Self-harm risk detected",
            matched_trigger=user_text[:60] if user_text else None,
        )

        display_status = escalation_result.get("display_card", "Supervisor escalation has been triggered in demo mode.")

        return {
            "detected_intent": "CRISIS_RELATED",
            "detected_emotion": "distressed",
            "selected_action": "CRISIS_RESPONSE",
            "tool_name": "crisis_support_protocol",
            "tool_result": crisis_text,
            "escalation_status": display_status,
            "reasoning_summary": "Deterministic safety guard detected HIGH risk indicators; normal agentic routing bypassed, crisis resources delivered, and human supervisor alert triggered.",
            "evaluation_result": "PASS",
            "final_response": crisis_text,
        }
    return crisis_node


# Alias for multimodal and lab 3 compatibility
create_crisis_escalation_node = create_crisis_node


def create_intent_node(llm=None, use_simulation: bool = False):
    """Create intent classification node."""
    _llm = None if use_simulation else (llm or get_llm())
    def intent_node(state: AgentState) -> Dict[str, Any]:
        intent = classify_intent(state.get("user_input", ""), llm=_llm)
        return {"detected_intent": intent}
    return intent_node


def create_emotion_node(llm=None, use_simulation: bool = False):
    """Create emotion analysis node."""
    _llm = None if use_simulation else (llm or get_llm())
    def emotion_node(state: AgentState) -> Dict[str, Any]:
        emotion = classify_emotion(state.get("user_input", ""), llm=_llm)
        return {"detected_emotion": emotion}
    return emotion_node


def create_planner_node():
    """Create agentic planner node adhering to hierarchical priorities 0-9."""
    def planner_node(state: AgentState) -> Dict[str, Any]:
        decision = plan_action(
            user_input=state.get("user_input", ""),
            detected_intent=state.get("detected_intent", "GENERAL_CONVERSATION"),
            detected_emotion=state.get("detected_emotion", "neutral"),
            risk_level=state.get("risk_level", "LOW"),
            safety_status=state.get("safety_status"),
            health_priority=state.get("health_priority"),
            input_type=state.get("input_type", "text"),
            image_present=bool(state.get("image_data")),
            previous_action=state.get("selected_action"),
            evaluation_result=state.get("evaluation_result"),
            retry_count=state.get("retry_count", 0),
        )
        return {
            "selected_action": decision["selected_action"],
            "tool_name": decision["tool_name"],
            "reasoning_summary": decision["reasoning_summary"],
        }
    return planner_node


def create_grounding_node():
    """Create grounding tool execution node."""
    def grounding_node(state: AgentState) -> Dict[str, Any]:
        try:
            output = interactive_grounding_tool(
                activity_type=state.get("tool_name"),
                emotion=state.get("detected_emotion", "stressed"),
            )
        except Exception as exc:
            output = f"Error: tool failed - {str(exc)}"
        return {"tool_result": output}
    return grounding_node


def create_coping_node(coping_llm=None, use_simulation: bool = False):
    """Create coping tool execution node."""
    _llm = None if use_simulation else (coping_llm or get_llm(model=settings.coping_model))
    def coping_node(state: AgentState) -> Dict[str, Any]:
        try:
            output = generate_coping_strategies(
                user_text=state.get("user_input", ""),
                emotion=state.get("detected_emotion", "stressed"),
                llm=_llm,
            )
        except Exception as exc:
            output = f"Error: tool failed - {str(exc)}"
        return {"tool_result": output}
    return coping_node


def create_rag_node(llm=None, use_simulation: bool = False):
    """Create Gale Encyclopedia of Medicine RAG execution node."""
    _llm = None if use_simulation else (llm or get_llm())
    def rag_node(state: AgentState) -> Dict[str, Any]:
        user_text = state.get("user_input", "")

        # Priority 2: Potentially acute / urgent medical handling
        if state.get("tool_name") == "urgent_medical_protocol" or state.get("health_priority") == "URGENT":
            urgent_res = screen_urgent_medical(user_text)
            advisory = urgent_res.get("advisory") or (
                "🚨 **Urgent Medical Assessment Recommended**:\n"
                "Severe, rapidly worsening, or acute physical symptoms require immediate professional clinical evaluation. "
                "Please contact an emergency department, urgent care clinic, or call local emergency medical services right away.\n\n"
                "*(Mindly is an educational wellness companion and does not provide clinical triage or medical diagnosis.)*"
            )
            return {
                "tool_result": advisory,
                "retrieved_context": advisory,
                "retrieved_sources": [{"source": "Mindly Clinical Triage Guidance", "page": 1, "topic": "Urgent Medical Evaluation"}],
            }

        try:
            answer, chunks = execute_rag_query(
                query=user_text,
                llm=_llm,
                top_k=3,
                detected_emotion=state.get("detected_emotion"),
            )
            return {
                "tool_result": answer,
                "retrieved_context": answer,
                "retrieved_sources": chunks,
            }
        except Exception as exc:
            return {
                "tool_result": f"Error: tool failed - {str(exc)}",
                "retrieved_context": None,
                "retrieved_sources": [],
            }
    return rag_node


def create_vision_node():
    """Create multimodal vision analysis node."""
    def vision_node(state: AgentState) -> Dict[str, Any]:
        try:
            image_input = state.get("image_data")
            user_prompt = state.get("user_input", "")
            res = analyze_image(image_input=image_input, prompt=user_prompt)
            output = res.get("summary", "Vision processing completed.")
            return {
                "tool_result": output,
                "vision_result": output,
            }
        except Exception as exc:
            return {"tool_result": f"Error: tool failed - {str(exc)}"}
    return vision_node


def create_journal_node():
    """Create wellness reflection journaling node."""
    def journal_node(state: AgentState) -> Dict[str, Any]:
        try:
            res = handle_journal_command(
                text=state.get("user_input", ""),
                mood=state.get("detected_emotion", "reflective"),
            )
            return {"tool_result": res, "journal_result": res}
        except Exception as exc:
            return {"tool_result": f"Error: tool failed - {str(exc)}"}
    return journal_node


def create_goal_node():
    """Create goals & habit tracking node."""
    def goal_node(state: AgentState) -> Dict[str, Any]:
        try:
            res = handle_goal_command(text=state.get("user_input", ""))
            return {"tool_result": res, "goal_context": res}
        except Exception as exc:
            return {"tool_result": f"Error: tool failed - {str(exc)}"}
    return goal_node


def create_memory_node():
    """Create long-term memory vault node."""
    def memory_node(state: AgentState) -> Dict[str, Any]:
        try:
            res = handle_memory_command(text=state.get("user_input", ""))
            return {"tool_result": res, "memory_context": res}
        except Exception as exc:
            return {"tool_result": f"Error: tool failed - {str(exc)}"}
    return memory_node


def create_direct_response_node(llm=None, use_simulation: bool = False):
    """Create direct conversational response node."""
    _llm = None if use_simulation else (llm or get_llm())
    def direct_response_node(state: AgentState) -> Dict[str, Any]:
        user_text = state.get("user_input", "")

        # Priority 4: Educational or Third-Person crisis support protocol
        if state.get("tool_name") == "third_person_support_protocol" or state.get("safety_status") == "EDUCATIONAL_OR_THIRD_PERSON":
            output = generate_third_person_support_response(user_text)
            return {"tool_result": output}

        try:
            output = generate_direct_response(
                user_text=user_text,
                messages=list(state.get("messages", [])),
                llm=_llm,
            )
        except Exception as exc:
            output = f"Error: tool failed - {str(exc)}"
        return {"tool_result": output}
    return direct_response_node


def create_evaluator_node():
    """Create evaluator node with retry count tracking and safety validation."""
    def evaluator_node(state: AgentState) -> Dict[str, Any]:
        current_retries = state.get("retry_count", 0)
        eval_dict = evaluate_execution(
            user_input=state.get("user_input", ""),
            selected_action=state.get("selected_action", "DIRECT_RESPONSE"),
            tool_result=state.get("tool_result"),
            retry_count=current_retries,
            max_retries=settings.max_eval_retries,
            risk_level=state.get("risk_level", "LOW"),
            escalation_status=state.get("escalation_status"),
            tool_name=state.get("tool_name"),
        )
        new_retries = current_retries + (1 if eval_dict["evaluation_result"] == "RETRY" else 0)
        return {
            "evaluation_result": eval_dict["evaluation_result"],
            "evaluation_feedback": eval_dict.get("evaluation_feedback"),
            "retry_count": new_retries,
            "replan_count": new_retries,
        }
    return evaluator_node


def create_final_response_node():
    """Create final response node assembling state for user presentation."""
    def final_response_node(state: AgentState) -> Dict[str, Any]:
        response_text = state.get("tool_result") or "I am here with you. How can I support you right now?"

        # Safe error sanitization
        if "error: tool failed" in response_text.lower():
            response_text = (
                "I am here with you, but I experienced a quiet moment of technical hesitation while processing that request. "
                "Take a slow breath, and let me know if you would like to try a different approach or simply talk through how you are feeling."
            )
        
        # If MEDIUM risk (acute distress without harm), append supportive note
        if state.get("risk_level") == "MEDIUM" and state.get("selected_action") != "CRISIS_RESPONSE":
            disclaimer = "\n\n*(If you are feeling deeply overwhelmed, remember that support is always available and you never have to carry this alone.)*"
            if disclaimer not in response_text:
                response_text = response_text + disclaimer

        return {
            "final_response": response_text,
            "messages": [AIMessage(content=response_text)],
        }
    return final_response_node


# ------------------------------------------------------------------------------
# CONDITIONAL ROUTING FUNCTIONS
# ------------------------------------------------------------------------------
def route_safety(state: AgentState) -> str:
    """Route based on deterministic safety assessment."""
    if state.get("risk_level") == "HIGH":
        return "crisis_node"
    return "intent_node"


def route_action(state: AgentState) -> str:
    """Route from planner to the selected tool node with safe fallbacks."""
    action = state.get("selected_action", "DIRECT_RESPONSE")
    if action == "USE_GROUNDING_TOOL":
        return "grounding_node"
    elif action == "USE_COPING_TOOL":
        return "coping_node"
    elif action == "USE_RAG_TOOL":
        return "rag_node"
    elif action == "USE_VISION_TOOL":
        return "vision_node"
    elif action == "USE_JOURNAL_TOOL":
        return "journal_node"
    elif action == "USE_GOAL_TOOL":
        return "goal_node"
    elif action == "USE_MEMORY_TOOL":
        return "memory_node"
    elif action == "CRISIS_RESPONSE":
        return "crisis_node"
    return "direct_response_node"


def route_evaluation(state: AgentState) -> str:
    """Route from evaluator back to planner if retry needed, else to final response."""
    if (
        state.get("risk_level") != "HIGH"
        and state.get("evaluation_result") == "RETRY"
        and state.get("retry_count", 0) < settings.max_eval_retries
    ):
        return "planner_node"
    return "final_response_node"


# ------------------------------------------------------------------------------
# GRAPH BUILDER
# ------------------------------------------------------------------------------
def build_mindly_graph(
    checkpointer: Optional[MemorySaver] = None,
    llm=None,
    coping_llm=None,
    use_simulation: bool = False,
):
    """Build and compile the complete Mindly Lab 3 LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # 1. Add all nodes
    builder.add_node("safety_guard_node", create_safety_guard_node())
    builder.add_node("crisis_node", create_crisis_node())
    builder.add_node("intent_node", create_intent_node(llm=llm, use_simulation=use_simulation))
    builder.add_node("emotion_node", create_emotion_node(llm=llm, use_simulation=use_simulation))
    builder.add_node("planner_node", create_planner_node())
    builder.add_node("grounding_node", create_grounding_node())
    builder.add_node("coping_node", create_coping_node(coping_llm=coping_llm, use_simulation=use_simulation))
    builder.add_node("rag_node", create_rag_node(llm=llm, use_simulation=use_simulation))
    builder.add_node("vision_node", create_vision_node())
    builder.add_node("journal_node", create_journal_node())
    builder.add_node("goal_node", create_goal_node())
    builder.add_node("memory_node", create_memory_node())
    builder.add_node("direct_response_node", create_direct_response_node(llm=llm, use_simulation=use_simulation))
    builder.add_node("evaluator_node", create_evaluator_node())
    builder.add_node("final_response_node", create_final_response_node())

    # 2. Add edges
    builder.add_edge(START, "safety_guard_node")

    # Conditional edge: Safety check
    builder.add_conditional_edges(
        "safety_guard_node",
        route_safety,
        {
            "crisis_node": "crisis_node",
            "intent_node": "intent_node",
        },
    )

    # Analysis sequence
    builder.add_edge("intent_node", "emotion_node")
    builder.add_edge("emotion_node", "planner_node")

    # Conditional edge: Dynamic Action Routing
    builder.add_conditional_edges(
        "planner_node",
        route_action,
        {
            "grounding_node": "grounding_node",
            "coping_node": "coping_node",
            "rag_node": "rag_node",
            "vision_node": "vision_node",
            "journal_node": "journal_node",
            "goal_node": "goal_node",
            "memory_node": "memory_node",
            "direct_response_node": "direct_response_node",
            "crisis_node": "crisis_node",
        },
    )

    # Tool convergence to Evaluator
    builder.add_edge("grounding_node", "evaluator_node")
    builder.add_edge("coping_node", "evaluator_node")
    builder.add_edge("rag_node", "evaluator_node")
    builder.add_edge("vision_node", "evaluator_node")
    builder.add_edge("journal_node", "evaluator_node")
    builder.add_edge("goal_node", "evaluator_node")
    builder.add_edge("memory_node", "evaluator_node")
    builder.add_edge("direct_response_node", "evaluator_node")

    # Section 6: High-risk crisis node also passes through safety evaluation
    builder.add_edge("crisis_node", "evaluator_node")

    # Conditional edge: Evaluation & Replanning Loop
    builder.add_conditional_edges(
        "evaluator_node",
        route_evaluation,
        {
            "planner_node": "planner_node",
            "final_response_node": "final_response_node",
        },
    )

    # Final response to END
    builder.add_edge("final_response_node", END)

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=memory)


# Singletons
_shared_checkpointer = MemorySaver()
_shared_live_graph = None
_shared_sim_graph = None


def get_mindly_graph(reload: bool = False, use_simulation: bool = False):
    """Retrieve or initialize compiled graph with thread checkpointer."""
    global _shared_live_graph, _shared_sim_graph
    if use_simulation:
        if _shared_sim_graph is None or reload:
            _shared_sim_graph = build_mindly_graph(checkpointer=_shared_checkpointer, use_simulation=True)
        return _shared_sim_graph
    else:
        if _shared_live_graph is None or reload:
            _shared_live_graph = build_mindly_graph(checkpointer=_shared_checkpointer, use_simulation=False)
        return _shared_live_graph


def send_message(
    user_text: str,
    thread_id: str,
    input_type: str = "text",
    image_data: Optional[Any] = None,
    graph=None,
    use_simulation: bool = False,
) -> Dict[str, Any]:
    """Execute Mindly multimodal agentic pipeline for a user message."""
    clean_text = (user_text or "").strip()

    # Empty text guard (allowed if image is provided)
    if not clean_text and not image_data:
        return {
            "response": "I am here with you whenever you are ready to share. How can I support you right now?",
            "intent": "GENERAL_CONVERSATION",
            "emotion": "neutral",
            "risk_level": "LOW",
            "action": "DIRECT_RESPONSE",
            "tool": "direct_conversation",
            "reasoning": "Received empty input; responded with gentle welcoming invitation.",
            "evaluation": "PASS",
            "retries": 0,
            "sources": [],
            "escalation": None,
        }

    # Context bounds
    MAX_INPUT_LEN = 3000
    if len(clean_text) > MAX_INPUT_LEN:
        clean_text = clean_text[:MAX_INPUT_LEN] + "..."

    active_graph = graph or get_mindly_graph(use_simulation=use_simulation)
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "user_input": clean_text,
        "input_type": input_type,
        "image_data": image_data,
        "messages": [HumanMessage(content=clean_text or "[Image Uploaded]")],
        "retry_count": 0,
    }

    try:
        result = active_graph.invoke(input_state, config=config)
    except Exception as exc:
        exc_str = str(exc).lower()
        if any(term in exc_str for term in ["connection", "refused", "timeout", "offline", "connecterror"]):
            sim_graph = get_mindly_graph(use_simulation=True)
            result = sim_graph.invoke(input_state, config=config)
            note = "\n\n*(Mindly cannot connect to the local AI model. Please make sure Ollama is running (`ollama serve`). Running in gentle offline practice mode.)*"
            if note not in result.get("final_response", ""):
                result["final_response"] = (result.get("final_response") or "") + note
        else:
            raise exc

    final_resp = result.get("final_response")
    if not final_resp:
        messages = result.get("messages", [])
        if messages:
            final_resp = str(messages[-1].content)
        else:
            final_resp = "I am here with you. How can I support you right now?"

    return {
        "response": final_resp,
        "intent": result.get("detected_intent", "GENERAL_CONVERSATION"),
        "emotion": result.get("detected_emotion", "neutral"),
        "risk_level": result.get("risk_level", "LOW"),
        "safety_status": result.get("safety_status", "PASSED"),
        "health_priority": result.get("health_priority", "NONE"),
        "action": result.get("selected_action", "DIRECT_RESPONSE"),
        "tool": result.get("tool_name", "direct_conversation"),
        "reasoning": result.get("reasoning_summary", "Selected direct empathetic response."),
        "evaluation": result.get("evaluation_result", "PASS"),
        "retries": result.get("retry_count", 0),
        "replan_count": result.get("replan_count", 0),
        "sources": result.get("retrieved_sources", []),
        "escalation": result.get("escalation_status"),
        "vision_result": result.get("vision_result"),
    }
