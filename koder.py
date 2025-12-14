#!/usr/bin/env python3
"""
Basic Coding Agent - Single file implementation using only standard library

This script demonstrates the core principles of a coding agent:
- Takes user prompts
- Provides project context
- Allows file reading/writing
- Uses OpenAI-compatible API for LLM calls
- Implements basic tool usage pattern
- Session context management for conversation history

Usage:
    python basic_coding_agent.py [--model MODEL_NAME] [--session SESSION_NAME] [--load-session]

Requirements:
    - Python 3.7+
    - OpenAI-compatible API endpoint
    - OPENAI_API_KEY environment variable
"""

import os
import json
import re
import urllib.request
import urllib.error
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime
import hashlib


# Configuration
@dataclass
class Config:
    api_base: str = "https://api.openai.com/v1"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    model: str = "gpt-4"
    model: str = "nex-agi/deepseek-v3.1-nex-n1:free"
    max_tokens: int = 2000
    temperature: float = 0.2
    max_context_files: int = 20
    max_file_size: int = 100000  # 100KB
    session_name: Optional[str] = None
    sessions_dir: Path = Path.home() / ".basic_coding_agent" / "sessions"


# Data structures
@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    function: Dict[str, Any]
    type: str = "function"


@dataclass
class LLMResponse:
    choices: List[Dict]
    usage: Dict[str, int]


@dataclass
class SessionContext:
    session_id: str
    created_at: str
    last_accessed: str
    message_count: int
    total_tokens: int
    project_root: str
    recent_summary: str


class BasicCodingAgent:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.messages: List[Message] = []
        self.project_root = Path.cwd()
        self.session_context: Optional[SessionContext] = None
        self.conversation_summary = ""
        self.total_tokens_used = 0

        # Setup sessions directory
        self.config.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session if requested
        if config.session_name:
            self._load_or_create_session(config.session_name)

        # Initialize with system prompt
        self._initialize_system_prompt()

    def _load_or_create_session(self, session_name: str):
        """Load existing session or create new one"""
        session_file = self.config.sessions_dir / f"{session_name}.json"

        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                # Load session context
                context_data = session_data.get('context', {})
                self.session_context = SessionContext(
                    session_id=context_data.get('session_id', session_name),
                    created_at=context_data.get('created_at', datetime.now().isoformat()),
                    last_accessed=datetime.now().isoformat(),
                    message_count=context_data.get('message_count', 0),
                    total_tokens=context_data.get('total_tokens', 0),
                    project_root=str(self.project_root),
                    recent_summary=context_data.get('recent_summary', '')
                )

                # Load conversation summary
                self.conversation_summary = session_data.get('conversation_summary', '')
                self.total_tokens_used = session_data.get('total_tokens', 0)

                # Load previous messages (limited to recent ones for context)
                saved_messages = session_data.get('messages', [])
                if saved_messages:
                    # Only load last few messages to avoid context overflow
                    recent_messages = saved_messages[-10:]  # Load last 10 messages
                    for msg_data in recent_messages:
                        self.messages.append(Message(
                            role=msg_data['role'],
                            content=msg_data['content'],
                            timestamp=msg_data.get('timestamp')
                        ))

                print(f"📂 Loaded session '{session_name}' ({len(saved_messages)} previous messages)")

            except Exception as e:
                print(f"⚠️  Could not load session: {e}. Starting fresh session.")
                self._create_new_session(session_name)
        else:
            self._create_new_session(session_name)

    def _create_new_session(self, session_name: str):
        """Create a new session"""
        self.session_context = SessionContext(
            session_id=session_name,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            message_count=0,
            total_tokens=0,
            project_root=str(self.project_root),
            recent_summary=''
        )
        print(f"📝 Created new session '{session_name}'")

    def _save_session(self):
        """Save current session to disk"""
        if not self.session_context:
            return

        try:
            session_file = self.config.sessions_dir / f"{self.session_context.session_id}.json"

            # Update session context
            self.session_context.last_accessed = datetime.now().isoformat()
            self.session_context.message_count = len(self.messages)
            self.session_context.total_tokens = self.total_tokens_used
            self.session_context.recent_summary = self._generate_conversation_summary()

            # Prepare session data
            session_data = {
                'context': {
                    'session_id': self.session_context.session_id,
                    'created_at': self.session_context.created_at,
                    'last_accessed': self.session_context.last_accessed,
                    'message_count': self.session_context.message_count,
                    'total_tokens': self.session_context.total_tokens,
                    'project_root': self.session_context.project_root,
                    'recent_summary': self.session_context.recent_summary
                },
                'conversation_summary': self.conversation_summary,
                'total_tokens': self.total_tokens_used,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp or datetime.now().isoformat()
                    }
                    for msg in self.messages
                ]
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️  Could not save session: {e}")

    def _generate_conversation_summary(self) -> str:
        """Generate a brief summary of recent conversation for context"""
        if not self.messages:
            return ""

        # Take last few user-assistant exchanges
        recent_exchanges = []
        for i in range(max(0, len(self.messages) - 6), len(self.messages)):
            msg = self.messages[i]
            if msg.role in ['user', 'assistant'] and len(msg.content) < 200:
                role = "User" if msg.role == "user" else "Assistant"
                recent_exchanges.append(f"{role}: {msg.content[:100]}...")

        return " | ".join(recent_exchanges[-3:])  # Last 3 exchanges

    def _initialize_system_prompt(self):
        """Create system prompt with project and session context"""
        project_info = self._get_project_info()
        session_info = self._get_session_info()

        system_prompt = f"""You are a coding assistant helping with a Python project.

Project Context:
{project_info}

{session_info}

Available Tools:
1. read_file(path: str) -> str: Read file content
2. write_file(path: str, content: str) -> str: Write content to file
3. list_files(path: str = ".") -> List[str]: List files in directory

Rules:
- Always use tools when possible instead of suggesting code changes
- Be precise with file paths
- Handle errors gracefully
- Ask for clarification if needed
- Remember the conversation context and build upon previous interactions
"""

        self.messages.append(Message(role="system", content=system_prompt))

    def _get_session_info(self) -> str:
        """Get session-specific information for context"""
        if not self.session_context:
            return ""

        session_info = []
        session_info.append(f"Session Context:")
        session_info.append(f"  Session: {self.session_context.session_id}")
        session_info.append(f"  Created: {self.session_context.created_at.split('T')[0]}")
        session_info.append(f"  Messages: {self.session_context.message_count}")
        session_info.append(f"  Total Tokens: {self.session_context.total_tokens}")

        if self.conversation_summary:
            session_info.append(f"  Recent: {self.conversation_summary}")

        return "\n".join(session_info)

    def _get_project_info(self) -> str:
        """Get basic project information"""
        info = []

        # Project structure
        try:
            files = []
            for item in self.project_root.iterdir():
                if item.is_file() and item.suffix in ['.py', '.md', '.txt', '.json']:
                    files.append(f"  - {item.name}")
                elif item.is_dir() and not item.name.startswith('.'):
                    files.append(f"  - {item.name}/")

            if files:
                info.append("Project Structure:")
                info.extend(files[:self.config.max_context_files])
        except Exception as e:
            info.append(f"Could not read project structure: {e}")

        # Git info if available
        try:
            if (self.project_root / ".git").exists():
                info.append("\nGit Repository: Yes")
        except:
            pass

        return "\n".join(info) if info else "No project context available"

    def _call_llm(self, messages: List[Message]) -> LLMResponse:
        """Call OpenAI-compatible API"""
        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Prepare request
        payload = {
            "model": self.config.model,
            "messages": [self._message_to_dict(msg) for msg in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add tools if needed
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the content of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path"}
                        },
                        "required": []
                    }
                }
            }
        ]

        payload["tools"] = tools
        payload["tool_choice"] = "auto"

        # Make HTTP request
        url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        req = urllib.request.Request(url,
                                    data=json.dumps(payload).encode('utf-8'),
                                    headers=headers,
                                    method='POST')

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                return LLMResponse(
                    choices=data.get('choices', []),
                    usage=data.get('usage', {'prompt_tokens': 0, 'completion_tokens': 0})
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            raise Exception(f"API Error {e.code}: {error_body}")
        except Exception as e:
            raise Exception(f"Request failed: {e}")

    def _message_to_dict(self, message: Message) -> Dict:
        """Convert Message to dict for API"""
        result = {
            "role": message.role,
            "content": message.content
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in message.tool_calls
            ]

        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id

        if message.name:
            result["name"] = message.name

        return result

    def _parse_tool_calls(self, response: LLMResponse) -> List[ToolCall]:
        """Parse tool calls from LLM response"""
        tool_calls = []

        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.get('finish_reason') == 'tool_calls':
                for tc in choice.get('message', {}).get('tool_calls', []):
                    tool_calls.append(ToolCall(
                        id=tc['id'],
                        function=tc['function']
                    ))

        return tool_calls

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return result"""
        tool_name = tool_call.function['name']
        args = tool_call.function.get('arguments', '{}')

        try:
            args_dict = json.loads(args)
        except json.JSONDecodeError:
            return f"Error: Invalid tool arguments: {args}"

        try:
            if tool_name == "read_file":
                path = args_dict.get('path', '')
                return self._read_file(path)

            elif tool_name == "write_file":
                path = args_dict.get('path', '')
                content = args_dict.get('content', '')
                return self._write_file(path, content)

            elif tool_name == "list_files":
                path = args_dict.get('path', '.')
                return self._list_files(path)

            else:
                return f"Error: Unknown tool {tool_name}"

        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _read_file(self, path: str) -> str:
        """Read file content"""
        file_path = self._resolve_path(path)

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            return f"Error: File too large ({file_size} bytes, max {self.config.max_file_size})"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """Write content to file"""
        file_path = self._resolve_path(path)

        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        if file_path.exists():
            shutil.copy2(file_path, backup_path)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except Exception as e:
            # Restore backup if write failed
            if backup_path.exists():
                shutil.copy2(backup_path, file_path)
            return f"Error writing file: {e}"

    def _list_files(self, path: str = ".") -> str:
        """List files in directory"""
        dir_path = self._resolve_path(path)

        if not dir_path.is_dir():
            return f"Error: {dir_path} is not a directory"

        try:
            files = []
            for item in dir_path.iterdir():
                if item.is_file():
                    files.append(f"  - {item.name} ({item.stat().st_size} bytes)")
                elif item.is_dir():
                    files.append(f"  - {item.name}/")

            return "Files:\n" + "\n".join(sorted(files))
        except Exception as e:
            return f"Error listing files: {e}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve file path relative to project root"""
        if path.startswith('/'):
            return Path(path)
        return (self.project_root / path).resolve()

    def _add_tool_response(self, tool_call: ToolCall, result: str):
        """Add tool response to messages"""
        self.messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=tool_call.id,
            name=tool_call.function['name'],
            timestamp=datetime.now().isoformat()
        ))

    def run(self):
        """Main interaction loop"""
        print("Basic Coding Agent - Type 'quit' to exit")
        print(f"Project: {self.project_root}")
        print(f"Model: {self.config.model}")
        if self.session_context:
            print(f"Session: {self.session_context.session_id}")
        print()

        while True:
            try:
                # Get user input
                user_input = input(">> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    if self.session_context:
                        self._save_session()
                        print(f"💾 Session '{self.session_context.session_id}' saved")
                    break

                if not user_input:
                    continue

                # Add user message with timestamp
                self.messages.append(Message(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now().isoformat()
                ))

                # Call LLM
                response = self._call_llm(self.messages)

                # Parse response
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]

                    # Check for tool calls
                    tool_calls = self._parse_tool_calls(response)

                    if tool_calls:
                        print(f"\n🔧 Executing {len(tool_calls)} tool(s)...")

                        # Execute tools and add responses
                        for tool_call in tool_calls:
                            tool_name = tool_call.function['name']
                            result = self._execute_tool(tool_call)
                            
                            # Show concise tool results
                            if tool_name == "read_file":
                                # For read_file, show summary instead of full content
                                lines = result.split('\n')
                                if len(lines) > 10 or len(result) > 500:
                                    print(f"  📄 {tool_name}: {len(lines)} lines, {len(result)} bytes")
                                else:
                                    print(f"  📄 {tool_name}: {len(lines)} lines")
                            elif tool_name == "write_file":
                                print(f"  ✏️  {tool_name}: {result}")
                            elif tool_name == "list_files":
                                # Show summary for directory listings
                                lines = result.split('\n')
                                if len(lines) > 15:
                                    print(f"  📁 {tool_name}: {len(lines)-1} items")
                                else:
                                    print(f"  📁 {tool_name}: {len(lines)-1} items")

                            self._add_tool_response(tool_call, result)

                        # Call LLM again with tool results
                        print("\n🤖 Getting final response from LLM...")
                        response = self._call_llm(self.messages)
                        choice = response.choices[0]

                    # Get final response
                    assistant_message = choice['message']['content']
                    self.messages.append(Message(
                        role="assistant",
                        content=assistant_message,
                        timestamp=datetime.now().isoformat()
                    ))

                    print(f"\n🤖 {assistant_message}")

                    # Update token usage
                    usage = response.usage
                    self.total_tokens_used += usage['prompt_tokens'] + usage['completion_tokens']

                    print(f"\n💰 Tokens: {usage['prompt_tokens']} + {usage['completion_tokens']} = {usage['prompt_tokens'] + usage['completion_tokens']} (session: {self.total_tokens_used})")

                    # Auto-save session periodically
                    if self.session_context and len(self.messages) % 5 == 0:
                        self._save_session()

            except KeyboardInterrupt:
                if self.session_context:
                    self._save_session()
                    print(f"\n💾 Session '{self.session_context.session_id}' saved")
                print("Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                # Don't break on errors, continue the loop


def main():
    """Entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basic Coding Agent')
    parser.add_argument('--model', type=str,
                       help='Specify which model to use (e.g., gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--session', type=str,
                       help='Specify session name for context persistence')
    parser.add_argument('--load-session', action='store_true',
                       help='Load existing session if available')

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Get your API key from an OpenAI-compatible provider and set:")
        print("export OPENAI_API_KEY='your-key-here'")
        return

    # Create config with optional model override
    config = Config()
    if args.model:
        config.model = args.model
    if args.session:
        config.session_name = args.session

    # Create and run agent
    agent = BasicCodingAgent(config)
    agent.run()


if __name__ == "__main__":
    main()