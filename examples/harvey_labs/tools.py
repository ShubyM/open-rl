"""LAB tool adapters for tinker-cookbook tool environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult


@dataclass(frozen=True)
class LabTool:
  spec: dict[str, Any]
  executor: Any

  @property
  def name(self) -> str:
    return str(self.spec["name"])

  @property
  def description(self) -> str:
    return str(self.spec.get("description", ""))

  @property
  def parameters_schema(self) -> dict[str, Any]:
    return dict(self.spec.get("parameters", {"type": "object", "properties": {}}))

  def to_spec(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": self.parameters_schema,
    }

  async def run(self, input: ToolInput) -> ToolResult:
    result = self.executor.execute(self.name, input.arguments)
    return simple_tool_result(result, call_id=input.call_id or "", name=self.name)


def build_lab_tools(executor: Any, lab_tool_definitions: list[dict[str, Any]]) -> list[Tool]:
  tools: list[Tool] = []
  for spec in lab_tool_definitions:
    # Preserve Harvey LAB's canonical names. Its prompt and teacher traces
    # teach `read`; renaming only the live schema makes valid calls fail.
    tools.append(LabTool(spec=dict(spec), executor=executor))
  return tools
