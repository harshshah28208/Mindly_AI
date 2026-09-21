"""State definition for Mindly Lab 3 Multimodal LangGraph workflow.

Captures the complete agentic lifecycle:
UNDERSTAND -> ANALYZE -> DECIDE -> PLAN -> SELECT TOOL -> EXECUTE -> EVALUATE -> REPLAN -> RESPOND
"""

from typing import Annotated, Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Represents the multimodal agentic state across all LangGraph nodes."""

    # Message history & input modalities
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    normalized_input: str
    input_type: str  # 'text', 'voice', 'image'
    image_data: Optional[Any]
    voice_transcript: Optional[str]

    # Analysis, safety & routing
    detected_intent: str
    detected_emotion: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    safety_status: str  # 'PASSED', 'HIGH_RISK_TRIGGERED', 'MEDIUM_RISK_DISTRESS', 'EDUCATIONAL_OR_THIRD_PERSON'
    health_priority: str  # 'URGENT', 'ROUTINE', 'NONE'

    # Planning & tools
    selected_action: str  # 'DIRECT_RESPONSE', 'USE_GROUNDING_TOOL', 'USE_COPING_TOOL', 'USE_RAG_TOOL', etc.
    selected_tools: Optional[List[str]]
    tool_name: Optional[str]
    tool_result: Optional[str]
    reasoning_summary: str

    # Contextual augmentations
    retrieved_context: Optional[str]
    retrieved_sources: Optional[List[Dict[str, Any]]]
    vision_result: Optional[str]
    memory_context: Optional[str]
    journal_result: Optional[str]
    goal_context: Optional[str]

    # Evaluation & replanning
    evaluation_result: str  # 'PASS', 'RETRY'
    evaluation_feedback: Optional[str]
    retry_count: int
    replan_count: Optional[int]

    # Final outputs & escalation
    final_response: str
    audio_output_path: Optional[str]
    escalation_status: Optional[str]
