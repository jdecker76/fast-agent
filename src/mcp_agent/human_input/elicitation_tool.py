from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.types import Tool as McpTool
from pydantic import BaseModel, Field

from mcp_agent.core.constants import HUMAN_INPUT_TOOL_NAME
from mcp_agent.human_input.elicitation_handler import elicitation_input_callback
from mcp_agent.human_input.types import HumanInputRequest, HumanInputResponse

"""FormSpec models and tool builder for human elicitation."""


class OptionItem(BaseModel):
    value: Union[str, int, float, bool]
    label: Optional[str] = None


class FormField(BaseModel):
    name: str
    type: Literal["text", "textarea", "number", "checkbox", "radio"]
    label: Optional[str] = None
    help: Optional[str] = None
    default: Optional[Union[str, int, float, bool]] = None
    required: Optional[bool] = None
    # number constraints
    min: Optional[float] = None
    max: Optional[float] = None
    # select options (for radio)
    options: Optional[List[OptionItem]] = None


class HumanFormArgs(BaseModel):
    """Simplified form spec for human elicitation.

    Preferred shape for LLMs.
    """

    title: Optional[str] = None
    description: Optional[str] = None
    message: Optional[str] = None
    fields: List[FormField] = Field(default_factory=list, max_items=7)


def get_elicitation_tool() -> McpTool:
    """Build the MCP Tool schema for the elicitation-backed human input tool.

    Uses Pydantic models to derive a clean, portable JSON Schema suitable for providers.
    """
    # Build input schema from the Pydantic model
    # Build input schema from the simplified FormSpec
    schema = HumanFormArgs.model_json_schema()

    def _resolve_refs(fragment: Any, root: Dict[str, Any]) -> Any:
        """Inline $ref references within a JSON schema fragment using the given root schema.

        Supports local references of the form '#/$defs/Name'.
        """
        if not isinstance(fragment, dict):
            return fragment

        if "$ref" in fragment:
            ref_path: str = fragment["$ref"]
            if ref_path.startswith("#/$defs/") and "$defs" in root:
                key = (
                    ref_path.split("/#/$defs/")[-1]
                    if "/#/$defs/" in ref_path
                    else ref_path[len("#/$defs/") :]
                )
                # simple one-level lookup
                target = root.get("$defs", {}).get(key)
                if isinstance(target, dict):
                    # resolve refs within the target too
                    return _resolve_refs(target, root)
            # If cannot resolve, drop the $ref and fall back to object
            fragment = {k: v for k, v in fragment.items() if k != "$ref"}
            fragment.setdefault("type", "object")
            fragment.setdefault("properties", {})
            return fragment

        # Recurse into dict values and lists
        resolved: Dict[str, Any] = {}
        for k, v in fragment.items():
            if isinstance(v, dict):
                resolved[k] = _resolve_refs(v, root)
            elif isinstance(v, list):
                resolved[k] = [
                    _resolve_refs(item, root) if isinstance(item, (dict, list)) else item
                    for item in v
                ]
            else:
                resolved[k] = v
        return resolved

    # Coerce to the minimal shape expected by providers:
    # - Ensure top-level type is object
    # - Keep only properties/required/additionalProperties/title/description
    sanitized: Dict[str, Any] = {"type": "object"}
    if "properties" in schema:
        props = dict(schema["properties"])  # copy
        # Normalize property name if aliasing produced 'form_schema'
        if "form_schema" in props:
            props["schema"] = props.pop("form_schema")
        # Resolve any $ref in all properties (notably 'schema')
        props = _resolve_refs(props, schema)
        sanitized["properties"] = props
    else:
        sanitized["properties"] = {}
    if "required" in schema:
        sanitized["required"] = schema["required"]
    # Allow extra keys if needed; some models may add benign fields
    sanitized["additionalProperties"] = True

    return McpTool(
        name=HUMAN_INPUT_TOOL_NAME,
        description=(
            "Collect structured input from a human via a simple form. "
            "Provide up to 7 fields with types: text, textarea, number, checkbox, or radio. "
            "Each field may include label, help, default; numbers may include min/max; radio may include options (value/label). "
            "You may also add an optional message shown above the form."
        ),
        inputSchema=sanitized,
    )


async def run_elicitation_form(arguments: dict | str, agent_name: str | None = None) -> str:
    import json

    def parse_schema_string(val: str) -> dict | None:
        if not isinstance(val, str):
            return None
        s = val.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines)
        try:
            return json.loads(s)
        except Exception:
            return None

    if not isinstance(arguments, dict):
        if isinstance(arguments, str):
            parsed = parse_schema_string(arguments)
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                raise ValueError("Invalid arguments. Provide FormSpec or JSON Schema object.")
        else:
            raise ValueError("Invalid arguments. Provide FormSpec or JSON Schema object.")

    schema: dict | None = None
    message: str | None = None
    title: str | None = None
    description: str | None = None

    if isinstance(arguments.get("fields"), list):
        fields = arguments.get("fields")
        if len(fields) > 7:
            raise ValueError(
                f"Error: form requests {len(fields)} fields; the maximum allowed is 7."
            )

        properties: dict[str, Any] = {}
        required_fields: list[str] = []
        for field in fields:
            if not isinstance(field, dict):
                continue
            name = field.get("name")
            ftype = field.get("type")
            if not isinstance(name, str) or not isinstance(ftype, str):
                continue
            prop: dict[str, Any] = {}
            label = field.get("label")
            help_text = field.get("help")
            default = field.get("default")
            required_flag = field.get("required")

            if ftype in ("text", "textarea"):
                prop["type"] = "string"
            elif ftype == "number":
                prop["type"] = "number"
                if isinstance(field.get("min"), (int, float)):
                    prop["minimum"] = field.get("min")
                if isinstance(field.get("max"), (int, float)):
                    prop["maximum"] = field.get("max")
            elif ftype == "checkbox":
                prop["type"] = "boolean"
            elif ftype == "radio":
                prop["type"] = "string"
                options = field.get("options") or []
                enum_vals = []
                enum_names = []
                for opt in options:
                    if isinstance(opt, dict) and "value" in opt:
                        enum_vals.append(opt["value"])
                        if isinstance(opt.get("label"), str):
                            enum_names.append(opt.get("label"))
                    elif opt is not None:
                        enum_vals.append(opt)
                if enum_vals:
                    prop["enum"] = enum_vals
                    if enum_names and len(enum_names) == len(enum_vals):
                        prop["enumNames"] = enum_names
            else:
                continue

            desc_parts = []
            if isinstance(label, str) and label:
                desc_parts.append(label)
            if isinstance(help_text, str) and help_text:
                desc_parts.append(help_text)
            if desc_parts:
                prop["description"] = " - ".join(desc_parts)
            if default is not None:
                prop["default"] = default
            properties[name] = prop
            if isinstance(required_flag, bool) and required_flag:
                required_fields.append(name)

        if len(properties) == 0:
            raise ValueError("Invalid form specification: no valid fields provided.")

        schema = {"type": "object", "properties": properties}
        if required_fields:
            schema["required"] = required_fields

        title = arguments.get("title") if isinstance(arguments.get("title"), str) else None
        description = (
            arguments.get("description") if isinstance(arguments.get("description"), str) else None
        )
        msg = arguments.get("message")
        if isinstance(msg, str):
            message = msg
        if title:
            schema["title"] = title
        if description:
            schema["description"] = description

    elif isinstance(arguments.get("schema"), (dict, str)):
        schema = arguments.get("schema")
        if isinstance(schema, str):
            parsed = parse_schema_string(schema)
            if isinstance(parsed, dict):
                schema = parsed
            else:
                raise ValueError("Missing or invalid schema. Provide a JSON Schema object.")
        msg = arguments.get("message")
        if isinstance(msg, str):
            message = msg
        if isinstance(arguments.get("title"), str) and "title" not in schema:
            schema["title"] = arguments.get("title")
        if isinstance(arguments.get("description"), str) and "description" not in schema:
            schema["description"] = arguments.get("description")
        if isinstance(arguments.get("required"), list) and "required" not in schema:
            schema["required"] = arguments.get("required")
        if isinstance(arguments.get("properties"), dict) and "properties" not in schema:
            schema["properties"] = arguments.get("properties")

    elif ("type" in arguments and "properties" in arguments) or (
        "$schema" in arguments and "properties" in arguments
    ):
        schema = arguments
        message = None
    else:
        raise ValueError("Missing or invalid schema or fields in arguments.")

    props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    if len(props) > 7:
        raise ValueError(f"Error: schema requests {len(props)} fields; the maximum allowed is 7.")

    request = HumanInputRequest(
        prompt=message or schema.get("title") or "Please complete this form:",
        description=schema.get("description"),
        request_id=f"__human_input__{uuid.uuid4()}",
        metadata={
            "agent_name": agent_name or "Unknown Agent",
            "requested_schema": schema,
        },
    )

    response: HumanInputResponse = await elicitation_input_callback(
        request=request, agent_name=agent_name or "Unknown Agent", server_name="__human_input__"
    )

    return response.response if isinstance(response.response, str) else str(response.response)


def get_elicitation_fastmcp_tool() -> FastMCPTool:
    async def elicit(
        title: Optional[str] = None,
        description: Optional[str] = None,
        message: Optional[str] = None,
        fields: List[FormField] = Field(default_factory=list, max_items=7),
    ) -> str:
        args = {
            "title": title,
            "description": description,
            "message": message,
            "fields": [f.model_dump() if isinstance(f, BaseModel) else f for f in fields],
        }
        return await run_elicitation_form(args)

    tool = FastMCPTool.from_function(elicit)
    tool.name = HUMAN_INPUT_TOOL_NAME
    tool.description = (
        "Collect structured input from a human via a simple form. Provide up to 7 fields "
        "(text, textarea, number, checkbox, radio). Fields can include label, help, default; "
        "numbers support min/max; radio supports options (value/label); optional message is shown above the form."
    )
    # Harmonize input schema with the sanitized MCP schema for provider compatibility
    tool.parameters = get_elicitation_tool().inputSchema
    return tool
