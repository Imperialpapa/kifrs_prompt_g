"""
K-IFRS 1019 DBO Validation System - AI Interpretation Layer
============================================================
Backward-compatibility shim.
The actual implementation has been refactored into the `ai/` package.

Import path unchanged:
    from ai_layer import AIRuleInterpreter
"""

from ai.interpreter import AIRuleInterpreter  # noqa: F401

__all__ = ["AIRuleInterpreter"]
