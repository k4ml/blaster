# Koder - Basic Coding Agent

A lightweight, single-file coding agent implementation using only Python's standard library. Koder enables natural language interaction with your codebase through an OpenAI-compatible API.

## Overview

Koder is a command-line coding assistant that helps you work with code using natural language prompts. It provides intelligent file operations, project context awareness, and persistent session management - all in a single Python file with no external dependencies beyond standard library.

## Features

### 🚀 **Core Capabilities**
- **Natural Language Coding**: Interact with your codebase using plain English
- **File Operations**: Read, write, and explore files with intelligent context
- **Project Awareness**: Automatic project structure analysis and git integration
- **Session Management**: Persistent conversation history across sessions
- **Enhanced Input**: Full terminal editing with arrow keys, history, and shortcuts

### 🛠️ **Tool System**
- `read_file(path)`: Read file contents with size limits
- `write_file(path, content)`: Write files with automatic backups
- `list_files(path)`: Explore directory structures

### 💾 **Session Management**
- **Persistent History**: Conversations are saved and restored automatically
- **Context Summarization**: Intelligent conversation summaries for long sessions
- **Auto-save**: Periodic automatic saves during long conversations
- **Multiple Sessions**: Support for different project sessions

### ⌨️ **Enhanced Terminal Interface**
- **Line Editing**: Full cursor movement with arrow keys
- **History Navigation**: Scroll through command history with up/down arrows
- **Keyboard Shortcuts**:
  - `Ctrl+A`: Move to beginning of line
  - `Ctrl+E`: Move to end of line
  - `Ctrl+U`: Clear current line
  - `Ctrl+C`: Exit gracefully with session save

## Installation

### Prerequisites
- Python 3.7 or higher
- An OpenAI-compatible API endpoint
- API key from your provider

### Setup

1. **Clone or download** the `koder.py` file to your project directory

2. **Set up your API key** (choose one):

   ```bash
   # For OpenAI
   export OPENAI_API_KEY='your-openai-key-here'
   
   # For OpenRouter
   export OPENROUTER_API_KEY='your-openrouter-key-here'
   ```

3. **Make it executable** (optional):
   ```bash
   chmod +x koder.py
   ```

## Usage

### Basic Usage

```bash
# Start interactive mode
python koder.py

# Use a specific model
python koder.py --model gpt-4

# Start with a named session
python koder.py --session my-project
```

### Command Line Options

```bash
python koder.py [OPTIONS]

Options:
  --model MODEL        Specify which model to use (e.g., gpt-4, gpt-3.5-turbo)
  --session NAME       Specify session name for context persistence
  --load-session       Load existing session if available
```

### Interactive Commands

Once running, you can use these commands:
- `quit`, `exit`, `q`: Exit the agent (saves session if active)
- `Ctrl+C`: Keyboard interrupt (saves session if active)

### Example Interactions

```bash
>> Read the main.py file and explain what it does
🤖 I'll read the main.py file and explain its functionality.

🔧 Executing 1 tool(s)...
  📄 read_file: 45 lines

🤖 The main.py file contains a web server implementation using Flask...
```

```bash
>> List all Python files in the project
🤖 I'll list all Python files in your project directory.

🔧 Executing 1 tool(s)...
  📁 list_files: 12 items

🤖 Here are the Python files in your project:
  - main.py (1523 bytes)
  - utils.py (892 bytes)
  - models.py (2341 bytes)
```

```bash
>> Create a new file called config.json with basic settings
🤖 I'll create a new config.json file with basic settings.

🔧 Executing 1 tool(s)...
  ✏️  write_file: Successfully wrote 127 bytes to config.json

🤖 Created config.json with basic configuration settings including...
```

## Configuration

### Default Settings

Koder uses sensible defaults but can be customized by modifying the `Config` class in `koder.py`:

```python
@dataclass
class Config:
    api_base: str = "https://api.openai.com/v1"  # API endpoint
    model: str = "gpt-4"                          # Default model
    max_tokens: int = 2000                        # Response length limit
    temperature: float = 0.2                      # Creativity setting
    max_context_files: int = 20                   # Files shown in context
    max_file_size: int = 100000                   # Max file size (100KB)
```

### Supported API Providers

Koder works with any OpenAI-compatible API:

#### OpenAI
```python
Config.api_base = "https://api.openai.com/v1"
Config.model = "gpt-4"
# Set OPENAI_API_KEY environment variable
```

#### OpenRouter
```python
Config.api_base = "https://openrouter.ai/api/v1"
Config.model = "mistralai/mistral-7b-instruct"
# Set OPENROUTER_API_KEY environment variable
```

#### Local LLM Server
```python
Config.api_base = "http://localhost:8000/v1"
Config.model = "llama2-7b"
# No API key needed for local servers
```

## Session Management

### Session Files

Sessions are automatically saved to:
```
~/.basic_coding_agent/sessions/
├── my-project.json
├── documentation.json
└── bug-fixing.json
```

### Session Features

- **Automatic Loading**: Sessions are loaded when you specify the name
- **Incremental Saving**: Sessions are saved after every few interactions
- **Context Preservation**: Last 10 messages are preserved for continuity
- **Summary Generation**: Brief summaries help maintain conversation context

### Using Sessions

```bash
# Start a new session
python koder.py --session my-project

# Later, resume the same session
python koder.py --session my-project

# The agent will load previous conversation context
```

## Architecture

### Core Components

1. **BasicCodingAgent**: Main orchestrator class
   - Manages conversation flow
   - Handles tool execution
   - Maintains session state

2. **EnhancedInput**: Terminal input handler
   - Provides line editing capabilities
   - Maintains command history
   - Handles special keys and shortcuts

3. **Message System**: Conversation management
   - Stores message history
   - Handles tool calls and responses
   - Manages timestamps and metadata

4. **Tool System**: File operations
   - Read/write file operations
   - Directory exploration
   - Automatic backup creation

### Data Flow

```
User Input → Message History → LLM API → Tool Execution → Response → User
     ↓                                                            ↓
Session Save ←──────────────────────────────────────────── Token Tracking
```

## Advanced Usage

### Working with Large Projects

Koder intelligently manages project context:

- **Automatic Truncation**: Large directory listings are summarized
- **Size Limits**: Files over 100KB are not read (configurable)
- **Context Limits**: Only most relevant files shown in project overview

### File Safety

- **Automatic Backups**: Original files are backed up before writing (`.bak` extension)
- **Error Recovery**: Failed writes automatically restore from backup
- **Size Checks**: Large files are rejected to prevent memory issues

### Token Management

- **Usage Tracking**: Token counts tracked per request and session
- **Budget Awareness**: Running totals help monitor API costs
- **Efficient Context**: Only essential context included in requests

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY environment variable not set"**
```bash
export OPENAI_API_KEY='your-key-here'
# or
export OPENROUTER_API_KEY='your-key-here'
```

**"API Error 401: Invalid API key"**
- Verify your API key is correct
- Check if the key has sufficient credits
- Ensure you're using the correct environment variable name

**"File too large"**
- The default limit is 100KB per file
- Modify `max_file_size` in the Config class to adjust

**"Request failed: Connection timeout"**
- Check your internet connection
- Verify the API endpoint URL is correct
- Some networks may block certain API endpoints

### Performance Tips

1. **Use specific session names** for different projects
2. **Keep conversations focused** to maintain context efficiency
3. **Use clear, specific prompts** for better results
4. **Monitor token usage** to manage API costs

## Comparison with Mistral Vibe

This implementation (`koder.py`) is a lightweight, single-file alternative to the full Mistral Vibe agent implementation. Key differences:

| Feature | Koder (Basic) | Mistral Vibe (Full) |
|---------|---------------|---------------------|
| **Dependencies** | Standard library only | Multiple external packages |
| **File Count** | Single file | Full package structure |
| **Setup** | Copy and run | pip install + configuration |
| **Features** | Core functionality | Advanced tools, UI, plugins |
| **Flexibility** | Modify source | Configuration files |
| **Use Case** | Quick setup, learning | Production, team use |

## Development

### Extending Koder

To add new features, you can modify these key areas:

1. **Add New Tools**: Implement new functions in the tool system
2. **Customize Prompts**: Modify the system prompt generation
3. **Add Providers**: Extend API backend support
4. **Enhance UI**: Improve the terminal interface

### Code Structure

```python
# Main components to modify:
class BasicCodingAgent:      # Core logic
class EnhancedInput:          # Terminal interface  
class Config:                 # Configuration
class Message:                # Data structure
```

## Contributing

This is a basic implementation meant for learning and quick setup. To contribute:

1. **Keep it simple**: Maintain single-file design
2. **Standard library only**: No external dependencies
3. **Clear documentation**: Comment complex sections
4. **Backward compatibility**: Don't break existing functionality

## License

This implementation is provided as-is for educational and personal use. Please respect the API provider's terms of service and pricing policies.

## Support

For issues specific to this implementation:
- Check the troubleshooting section above
- Review the inline code comments
- Examine session files in `~/.basic_coding_agent/sessions/`

For API-related issues:
- OpenAI: https://platform.openai.com/docs/api-reference
- OpenRouter: https://openrouter.ai/docs

---

**Note**: This is a basic implementation demonstrating core concepts. For production use with advanced features, consider the full Mistral Vibe implementation documented in `MISTRAL_VIBE_AGENT_IMPLEMENTATION.md`.