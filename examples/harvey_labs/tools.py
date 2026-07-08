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
    lab_name: str | None = None
    should_stop: bool = False

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
        if self.should_stop:
            return simple_tool_result(
                f"Submitted: {input.arguments.get('note', 'done')}",
                call_id=input.call_id or "",
                name=self.name,
                should_stop=True,
            )

        result = self.executor.execute(self.lab_name or self.name, input.arguments)
        return simple_tool_result(result, call_id=input.call_id or "", name=self.name)


def build_lab_tools(executor: Any, lab_tool_definitions: list[dict[str, Any]]) -> list[Tool]:
    tools = []
    for spec in lab_tool_definitions:
        display_spec = dict(spec)
        lab_name = str(display_spec["name"])
        if lab_name == "read":
            display_spec["name"] = "read_document"
        tools.append(LabTool(spec=display_spec, executor=executor, lab_name=lab_name))
    tools.append(
        LabTool(
            spec={
                "name": "submit",
                "description": "Submit the episode for rubric grading after writing the deliverable.",
                "parameters": {
                    "type": "object",
                    "properties": {"note": {"type": "string", "description": "Short completion note."}},
                },
            },
            executor=executor,
            should_stop=True,
        )
    )
    return tools
