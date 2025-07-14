import random
import string
from fastmcp import FastMCP # Install fastmcp to use this server

mcp = FastMCP(
    name="Creative writing tools",
    instructions="""
    This is an MCP server with tools for creative writing.
    """,
    dependencies=[]
)

@mcp.tool
def reverse_string(text: str) -> str:
    """Reverses a string."""
    return text[::-1]

@mcp.tool
def capitalize_string(text: str) -> str:
    """Capitalizes a string."""
    return text.upper()

@mcp.tool
def lowercase_string(text: str) -> str:
    """Converts a string to lowercase."""
    return text.lower()

@mcp.tool
def random_string(length: int) -> str:
    """Generates a random string of a given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

@mcp.tool
def random_case_string(text: str) -> str:
    """Randomly capitalizes or lowercase each letter in a string."""
    return ''.join(random.choice([str.upper, str.lower])(c) for c in text)

@mcp.tool
def coding_camel_case(text: str) -> str:
    """Converts a string to camel case."""
    return text.title().replace(" ", "")

@mcp.tool
def coding_snake_case(text: str) -> str:
    """Converts a string to snake case."""
    return text.lower().replace(" ", "_")

@mcp.tool
def coding_kebab_case(text: str) -> str:
    """Converts a string to kebab case."""
    return text.lower().replace(" ", "-")



if __name__ == "__main__":
    # Run in stdio mode
    mcp.run()