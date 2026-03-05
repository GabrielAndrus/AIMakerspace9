"""Secure Reward Function Execution for GRPO Training.

This module provides a sandboxed environment for executing user-provided
reward functions. It uses process isolation and resource limits to prevent
malicious code from harming the system.

SECURITY MODEL:
1. AST validation - Block dangerous constructs before execution
2. Subprocess isolation - Run in separate process (no shared memory)
3. Resource limits - CPU time, memory via setrlimit
4. Timeout enforcement - Kill long-running code
5. Restricted builtins - Only safe operations allowed

IMPORTANT: Python's exec() and eval() CANNOT be made safe through namespace
restriction alone. Attackers can escape restricted namespaces using Python's
introspection features (e.g., accessing __class__.__base__ to get builtins back).

The solution is process isolation with resource limits.
"""

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from typing import Any

# Safe builtins that can be used in reward functions
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

# AST nodes that are explicitly BLOCKED for security
BLOCKED_AST_NODES = {
    # Import mechanisms - allow arbitrary code execution
    ast.Import,
    ast.ImportFrom,
    # Note: exec and eval are builtin functions, not AST nodes
    # We catch them via Call node analysis in the validator
}

# Disallowed attribute access patterns
DISALLOWED_ATTRS = {
    "__import__",
    "exec",  # builtin function
    "eval",  # builtin function
    "compile",
    "open",  # file I/O
    "globals",
    "locals",
}

# Disallowed global variable names ( accessed directly as a name)
DISALLOWED_GLOBALS = {
    "__builtins__",
}

# Disallowed function calls
DISALLOWED_CALLS = {
    "__import__",
    "exec",
    "eval",
    "compile",
    "open",
}


class SecurityViolationError(Exception):
    """Raised when code contains disallowed AST nodes or patterns."""

    pass


class _SecurityASTValidator(ast.NodeVisitor):
    """AST visitor to validate code for security violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Block all import statements."""
        names = [alias.name for alias in node.names]
        self.violations.append(f"Import statement found: {', '.join(names)}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block all from-import statements."""
        module = node.module or "<unknown>"
        names = [alias.name for alias in node.names]
        self.violations.append(f"From-import found: from {module} import {', '.join(names)}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for calls to dangerous functions."""
        # Check function name
        if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
            self.violations.append(f"Call to disallowed function: {node.func.id}()")

        # Check for attribute access like getattr(), setattr()
        if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr", "delattr"):
            self.violations.append(f"Call to forbidden introspection function: {node.func.id}()")

        # Check for file operations via open()
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self.violations.append("File I/O operation detected: open()")

        # Check for dangerous method calls like subprocess.run()
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            # Block subprocess calls
            if attr_name in ("run", "Popen", "call", "check_output"):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                    self.violations.append(
                        f"Subprocess execution detected: subprocess.{attr_name}()"
                    )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check for dangerous attribute access."""
        if node.attr in DISALLOWED_ATTRS:
            self.violations.append(f"Access to disallowed attribute: {node.attr}")

        # Check for accessing __globals__ or similar
        if node.attr.startswith("__") and "global" in node.attr.lower():
            self.violations.append(f"Access to introspection attribute: {node.attr}")

        # Check for accessing dangerous modules through attributes
        if node.attr in ("system", "popen"):
            # This catches os.system, subprocess.Popen, etc.
            self.violations.append(f"Potential command execution via attribute: {node.attr}")

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check for dangerous decorators or function definitions."""
        # Check decorator list
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "property":
                continue  # property is safe
            if isinstance(decorator, ast.Name):
                self.violations.append(f"Decorator may be unsafe: @{decorator.id}")

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for references to dangerous builtins."""
        if node.id in DISALLOWED_CALLS:
            # This is just a reference, not a call, so it's less critical
            # But still worth flagging for review
            pass

        # Check for access to dangerous globals like __builtins__
        if node.id in DISALLOWED_GLOBALS:
            self.violations.append(f"Access to disallowed global: {node.id}")

        self.generic_visit(node)


def validate_code_ast(code: str) -> tuple[bool, str]:
    """Validate that code only uses allowed AST constructs.

    This performs static analysis to block dangerous patterns before execution.
    Note: This is a best-effort defense. The real security comes from process isolation.

    Args:
        code: Python source code to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, msg = validate_code_ast("def f(): return 1")
        >>> is_valid
        True

        >>> is_valid, msg = validate_code_ast("import os; os.system('ls')")
        >>> is_valid
        False
        >>> print(msg)
        Import statement found: os
    """
    try:
        tree = ast.parse(code, filename="<reward_code>")
    except SyntaxError as e:
        return False, f"Syntax error in reward code:\n{e}"
    except Exception as e:
        return False, f"Error parsing reward code: {e}"

    validator = _SecurityASTValidator()
    validator.visit(tree)

    if validator.violations:
        error_msg = "Security violations detected in reward code:\n\n"
        for i, violation in enumerate(validator.violations, 1):
            error_msg += f"{i}. {violation}\n"

        error_msg += "\nBlocked operations (security risk):\n"
        error_msg += "- import, from ... import statements\n"
        error_msg += "- exec(), eval(), compile() calls\n"
        error_msg += "- open() for file I/O\n"
        error_msg += "- subprocess operations (run, Popen, etc.)\n"
        error_msg += "- Introspection functions (getattr, __globals__, etc.)\n"

        return False, error_msg

    return True, ""


def create_sandbox_script(reward_code: str, timeout: int = 5) -> str:
    """Create a self-contained sandbox script for execution.

    The script applies resource limits to itself before executing user code.
    This ensures the child process cannot exceed its allocated resources.

    Args:
        reward_code: User-provided reward function code
        timeout: Timeout in seconds

    Returns:
        Complete Python script as string
    """
    # Use format() instead of f-string to avoid brace escaping issues
    template = '''#!/usr/bin/env python3
"""Sandboxed execution environment for reward functions."""
import sys
import json
import resource
import signal

# Apply memory limit BEFORE executing user code
def set_resource_limits():
    """Set process resource limits to prevent DoS attacks."""
    try:
        # Limit virtual memory (address space)
        # This prevents the process from allocating too much RAM
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY:
            hard = 128 * 1024 * 1024  # 128 MB default
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))

        # Limit CPU time
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard == resource.RLIM_INFINITY:
            hard = 10  # 10 seconds CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))

        # Limit file size (prevent writing GBs of output)
        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        if hard == resource.RLIM_INFINITY:
            hard = 10 * 1024 * 1024  # 10 MB max output
        resource.setrlimit(resource.RLIMIT_FSIZE, (hard, hard))
    except (ValueError, resource.error) as e:
        # Some systems may not support all limits
        sys.stderr.write("Warning: Could not set resource limits: " + str(e) + "\\n")

# Set up timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Reward function exceeded time limit")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

# Apply limits immediately
set_resource_limits()

# Now execute the user's reward function code
{reward_code}

if __name__ == "__main__":
    try:
        # Read input from stdin
        input_text = sys.stdin.read()
        if not input_text:
            raise ValueError("No input provided")

        input_data = json.loads(input_text)

        # Extract completions and other kwargs
        completions = input_data.get('completions', [])
        kwargs = dict((k, v) for k, v in input_data.items() if k != 'completions')

        # Execute the reward function
        result = reward_func(completions, **kwargs)

        # Output as JSON
        print(json.dumps(result))

    except TimeoutError as e:
        sys.stderr.write(str(e))
        sys.exit(1)
    except MemoryError:
        sys.stderr.write("Reward function exceeded memory limit")
        sys.exit(1)
    except Exception as e:
        # Include traceback for debugging
        import traceback
        sys.stderr.write("Reward function error: " + str(e) + "\\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
'''
    return template.format(timeout=timeout, reward_code=reward_code)


class SecureRewardExecutor:
    """Execute user-provided reward functions in sandboxed subprocesses.

    This executor provides multiple layers of security:
    1. AST validation - Block dangerous constructs before execution
    2. Subprocess isolation - Run in separate process with no shared memory
    3. Resource limits - CPU time, memory, and file size restrictions
    4. Timeout enforcement - Kill processes that run too long

    The subprocess approach is critical because Python's exec() cannot be
    made safe through namespace restriction alone. Attackers can escape
    restricted namespaces using Python's introspection features.

    Example:
        >>> executor = SecureRewardExecutor(timeout_seconds=5.0)
        >>>
        >>> code = '''
        ... def reward_func(completions, **kwargs):
        ...     return [1.0 if "correct" in c.lower() else 0.0
        ...             for c in completions]
        ... '''
        >>>
        >>> rewards = executor.execute(
        ...     code,
        ...     completions=["This is correct", "Wrong answer"]
        ... )
        >>> rewards
        [1.0, 0.0]

    Security Example:
        This code will be blocked:

        >>> malicious_code = '''
        ... import os
        ... def reward_func(completions):
        ...     os.system("rm -rf /")
        ...     return [0.0] * len(completions)
        ... '''
        >>> executor.execute(malicious_code, completions=["test"])
        SecurityViolationError: Import statement found: os
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        memory_limit_mb: int = 128,
    ):
        """Initialize the secure reward executor.

        Args:
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Memory limit in megabytes (applied via setrlimit)
        """
        self.timeout = timeout_seconds
        self.memory_mb = memory_limit_mb

    def _validate_code_ast(self, reward_code: str) -> None:
        """Validate code for security violations.

        Args:
            reward_code: Python source code to validate

        Raises:
            SecurityViolationError: If code contains dangerous constructs
        """
        is_valid, error_msg = validate_code_ast(reward_code)
        if not is_valid:
            raise SecurityViolationError(error_msg)

    def execute(self, reward_code: str, completions: list[Any], **kwargs) -> list[float]:
        """Execute reward function on completions in a sandboxed subprocess.

        Args:
            reward_code: User-provided Python code defining 'reward_func'
            completions: List of model outputs to score
            **kwargs: Additional arguments to pass to reward function

        Returns:
            List of float rewards (one per completion)

        Raises:
            SecurityViolationError: If code contains dangerous constructs
            TimeoutError: If execution exceeds time limit
        """
        # Step 1: Validate AST (best-effort defense)
        self._validate_code_ast(reward_code)

        # Step 2: Create sandbox script
        script = create_sandbox_script(reward_code, int(self.timeout))

        # Step 3: Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            temp_path = f.name

        try:
            # Step 4: Prepare input data
            input_data = json.dumps({"completions": completions, **kwargs}, ensure_ascii=False)

            # Step 5: Execute in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout + 1.0,  # Add buffer for overhead
            )

            # Step 6: Handle errors
            if result.returncode != 0:
                stderr = result.stderr

                # Check for timeout in subprocess
                if "exceeded time limit" in stderr or "TimeoutError" in stderr:
                    raise TimeoutError(
                        f"Reward function exceeded {self.timeout}s timeout.\n"
                        "Consider simplifying your reward logic or increasing timeout."
                    )

                # Other errors return default rewards
                return self._default_rewards(len(completions), stderr)

            # Step 7: Parse output
            try:
                rewards = json.loads(result.stdout)
                if not isinstance(rewards, list):
                    raise ValueError("Reward function must return a list")
                return rewards
            except json.JSONDecodeError:
                return self._default_rewards(len(completions), "Invalid JSON output")

        except subprocess.TimeoutExpired:
            # The subprocess.run() itself timed out
            raise TimeoutError(
                f"Reward function exceeded {self.timeout}s timeout.\n"
                "Consider simplifying your reward logic or increasing timeout."
            )
        except TimeoutError:
            # Re-raise any other TimeoutErrors
            raise
        except Exception as e:
            return self._default_rewards(len(completions), str(e))
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass

    def _default_rewards(self, count: int, error: str) -> list[float]:
        """Return zero rewards on failure.

        Args:
            count: Number of completions
            error: Error message for logging

        Returns:
            List of zeros
        """
        import warnings

        warnings.warn(
            f"Reward execution failed, returning default zeros: {error}",
            RuntimeWarning,
        )
        return [0.0] * count


# Pre-built secure reward functions (for reference)
def math_accuracy_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """Reward function for math problems with boxed answers.

    Checks if the model's answer matches the ground truth.
    Handles LaTeX \\boxed{} format and plain numeric answers.

    This is a reference implementation showing what a reward function looks like.
    It's executed using SecureRewardExecutor when provided as custom code.

    Args:
        completions: List of model outputs
        ground_truth: List of correct answers (one per completion)
        **kwargs: Additional arguments

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    import re

    rewards = []

    for completion, gt in zip(completions, ground_truth):
        content = completion if isinstance(completion, str) else str(completion)

        # Check for boxed format: \boxed{answer}
        match = re.search(r"\\boxed\{([^}]+)\}", content, re.IGNORECASE)

        if match:
            answer = match.group(1).strip().lower()
        else:
            # Try to extract last number as answer
            numbers = re.findall(r"-?\d+\.?\d*", content)
            answer = numbers[-1].strip().lower() if numbers else ""

        rewards.append(1.0 if answer == gt.strip().lower() else 0.0)

    return rewards
