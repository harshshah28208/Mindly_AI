"""Unit tests for Mindly Lab 2 dynamic tool suite."""

import unittest
from langchain_core.messages import HumanMessage
from tools.coping import generate_coping_strategies
from tools.direct_response import generate_direct_response
from tools.grounding import (
    interactive_grounding_tool,
    run_54321_grounding,
    run_box_breathing,
    run_guided_reflection,
)


class TestMindlyTools(unittest.TestCase):
    """Test individual grounding, coping, and direct response tools."""

    def test_box_breathing(self):
        result = run_box_breathing()
        self.assertIn("4-4-4-4", result)
        self.assertIn("Inhale", result)
        self.assertIn("Exhale", result)

    def test_54321_grounding(self):
        result = run_54321_grounding()
        self.assertIn("5 things you can SEE", result)
        self.assertIn("4 things you can TOUCH", result)
        self.assertIn("3 things you can HEAR", result)
        self.assertIn("2 things you can SMELL", result)
        self.assertIn("1 thing you can TASTE", result)

    def test_guided_reflection(self):
        result = run_guided_reflection(emotion="stressed")
        self.assertIn("Guided Mindful Reflection", result)
        self.assertIn("within my control", result)

    def test_interactive_grounding_dispatcher(self):
        res1 = interactive_grounding_tool("box_breathing", emotion="anxious")
        self.assertIn("Box Breathing", res1)

        res2 = interactive_grounding_tool("54321_grounding", emotion="overwhelmed")
        self.assertIn("5-4-3-2-1", res2)

    def test_coping_strategies(self):
        academic_coping = generate_coping_strategies("How do I handle my exams and study workload?", emotion="stressed")
        self.assertIn("Academic Workload", academic_coping)
        self.assertIn("Micro-Start", academic_coping)

    def test_direct_response_identity(self):
        reply = generate_direct_response("What is your name?")
        self.assertIn("Mindly", reply)

    def test_direct_response_memory_retention(self):
        history = [
            HumanMessage(content="Hello! My name is Rahul."),
        ]
        reply = generate_direct_response("What is my name?", messages=history)
        self.assertIn("Rahul", reply)


if __name__ == "__main__":
    unittest.main()
