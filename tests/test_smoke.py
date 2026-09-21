"""Test Category A: Project Startup, Syntax, Imports, and Graph Integrity (A1 - A10)."""

import ast
import os
from pathlib import Path
import pytest

from config.settings import settings
from graph import build_mindly_graph


def test_a1_modules_import_successfully():
    """A1: Verify all key project modules import cleanly without exceptions."""
    import config.settings
    import agent.state
    import agent.safety
    import agent.intent
    import agent.emotion
    import agent.planner
    import agent.evaluator
    import agent.prompts
    import rag.vectorstore
    import rag.ingestion
    import rag.retriever
    import tools.grounding
    import tools.coping
    import tools.direct_response
    import tools.rag
    import tools.voice
    import tools.vision
    import tools.escalation
    import tools.journal
    import tools.goals
    import tools.memory
    import memory.database
    import graph
    assert True


def test_a2_no_syntax_errors():
    """A2: Parse all Python files across the codebase using AST to verify valid syntax."""
    root = Path(__file__).resolve().parent.parent
    py_files = list(root.glob("*.py")) + list(root.glob("*/*.py"))
    for py_file in py_files:
        if ".venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        code = py_file.read_text(encoding="utf-8")
        try:
            ast.parse(code, filename=str(py_file))
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {py_file}: {e}")


def test_a3_no_circular_imports():
    """A3: Verify there are no circular dependencies by importing in diverse sequences."""
    import sys
    # Re-importing core graph should resolve without RecursionError or ImportCycles
    from graph import build_mindly_graph
    from agent.planner import plan_action
    from agent.evaluator import evaluate_execution
    assert callable(build_mindly_graph)
    assert callable(plan_action)
    assert callable(evaluate_execution)


def test_a4_configuration_loads_correctly():
    """A4: Central settings object loads environment configuration with valid defaults."""
    assert settings.app_name == "Mindly"
    assert settings.ollama_base_url is not None
    assert settings.ollama_model is not None
    assert settings.max_eval_retries >= 1
    assert settings.escalation_mode == "demo"
    assert settings.database_path.name == "app.db"


def test_a5_env_example_contents():
    """A5: .env.example contains required configuration parameters."""
    env_example = Path(__file__).resolve().parent.parent / ".env.example"
    assert env_example.exists(), ".env.example must exist"
    content = env_example.read_text(encoding="utf-8")
    required_keys = [
        "APP_NAME",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "ESCALATION_MODE",
        "MAX_EVAL_RETRIES",
        "KNOWLEDGE_DIR",
        "DATABASE_PATH",
    ]
    for key in required_keys:
        assert key in content, f"Missing {key} in .env.example"


def test_a6_application_state_initialization():
    """A6: Verify database and directory structures can initialize without errors."""
    from memory.database import db
    assert db.db_path.exists(), "SQLite database file should exist"
    assert settings.knowledge_dir.exists(), "Knowledge dir should exist"
    assert settings.vector_db_dir.exists(), "Vector DB dir should exist"


def test_a7_graph_compiles():
    """A7: LangGraph StateGraph compiles successfully."""
    graph = build_mindly_graph(use_simulation=True)
    assert graph is not None


def test_a8_all_expected_nodes_exist():
    """A8: Verify all 15 functional nodes and boundary nodes exist in the graph."""
    graph = build_mindly_graph(use_simulation=True)
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "safety_guard_node",
        "crisis_node",
        "intent_node",
        "emotion_node",
        "planner_node",
        "grounding_node",
        "coping_node",
        "rag_node",
        "vision_node",
        "journal_node",
        "goal_node",
        "memory_node",
        "direct_response_node",
        "evaluator_node",
        "final_response_node",
        "__start__",
        "__end__",
    }
    for n in expected:
        assert n in nodes, f"Expected node '{n}' missing from StateGraph"


def test_a9_all_conditional_routes_exist():
    """A9: Verify conditional edges exist for safety, dynamic routing, and evaluation."""
    from graph import route_safety, route_action, route_evaluation
    # Safety route: HIGH -> crisis_node, LOW -> intent_node
    assert route_safety({"risk_level": "HIGH"}) == "crisis_node"
    assert route_safety({"risk_level": "LOW"}) == "intent_node"

    # Action route: All tools mapped
    assert route_action({"selected_action": "USE_RAG_TOOL"}) == "rag_node"
    assert route_action({"selected_action": "USE_VISION_TOOL"}) == "vision_node"
    assert route_action({"selected_action": "USE_JOURNAL_TOOL"}) == "journal_node"
    assert route_action({"selected_action": "USE_GOAL_TOOL"}) == "goal_node"
    assert route_action({"selected_action": "USE_MEMORY_TOOL"}) == "memory_node"
    assert route_action({"selected_action": "USE_GROUNDING_TOOL"}) == "grounding_node"
    assert route_action({"selected_action": "USE_COPING_TOOL"}) == "coping_node"
    assert route_action({"selected_action": "DIRECT_RESPONSE"}) == "direct_response_node"
    assert route_action({"selected_action": "CRISIS_RESPONSE"}) == "crisis_node"

    # Evaluation route: RETRY -> planner_node, PASS -> final_response_node
    assert route_evaluation({"evaluation_result": "RETRY", "retry_count": 0}) == "planner_node"
    assert route_evaluation({"evaluation_result": "PASS", "retry_count": 0}) == "final_response_node"
    assert route_evaluation({"evaluation_result": "RETRY", "retry_count": 2}) == "final_response_node"


def test_a10_tools_registered_correctly():
    """A10: Verify all tool execution functions are callable and registered."""
    from tools.grounding import interactive_grounding_tool
    from tools.coping import generate_coping_strategies
    from tools.direct_response import generate_direct_response
    from tools.rag import execute_rag_query
    from tools.voice import synthesize_speech
    from tools.vision import analyze_image
    from tools.escalation import escalation_service
    from tools.journal import handle_journal_command
    from tools.goals import handle_goal_command
    from tools.memory import handle_memory_command

    for fn in [
        interactive_grounding_tool,
        generate_coping_strategies,
        generate_direct_response,
        execute_rag_query,
        synthesize_speech,
        analyze_image,
        handle_journal_command,
        handle_goal_command,
        handle_memory_command,
    ]:
        assert callable(fn), f"Tool {fn} is not callable"
    assert callable(escalation_service.escalate)


def test_aa_agent_state_internal_consistency(simulated_graph):
    """Category AA: Verify agent state fields are internally consistent across execution."""
    config = {"configurable": {"thread_id": "test_state_consistency_thread"}}
    state = simulated_graph.invoke({
        "user_input": "What is hypertension according to medical literature?",
        "input_type": "text",
        "messages": [],
    }, config=config)

    assert state.get("selected_action") == "USE_RAG_TOOL"
    assert state.get("tool_name") in ["gale_retriever", "gale_encyclopedia_retriever"]
    assert state.get("retrieved_context") is not None
    assert isinstance(state.get("retrieved_sources"), list)
    assert len(state.get("retrieved_sources")) > 0
    assert state.get("evaluation_result") == "PASS"
    assert state.get("final_response") is not None


def test_ab_session_state_and_database_persistence(clean_test_user):
    """Category AB: Verify database state persistence across simulated UI sessions."""
    from memory.database import db
    # 1. Store memory and journal in session 1
    db.save_memory(clean_test_user, "theme_pref", "dark_mode", "preferences")
    db.add_journal(clean_test_user, "Evening reflection on session.", "calm")
    
    # 2. Query in simulated new session
    mems = db.get_memories(clean_test_user)
    assert len(mems) == 1
    assert mems[0]["memory_value"] == "dark_mode"

    journals = db.get_journals(clean_test_user)
    assert len(journals) == 1
    assert journals[0]["entry_text"] == "Evening reflection on session."

