# MicroAgent HelloWorld Demo

A simple "Hello, World!" project that demonstrates how to use the `microagent` library.

The demo involves an agent that attempts to build the string "Hello, World!" by querying an LLM one character at a time. It showcases the core components of the library, including:
- A custom `MicroAgent` implementation.
- Using the `MicroAgentExecutor` to run the agent.
- Tracking execution statistics like cost and API calls.

## Prerequisites

1.  **Python & `uv`**: Ensure you have Python and the `uv` package manager installed.
2.  **LLM API Key**: An active API key is required. The demo automatically checks for the `OPENAI_API_KEY` environment variable.

## Quick Start (with `run.sh`)

The fastest way to run the demo is with the provided shell script, which handles all setup for you.

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Make the script executable (if you haven't already)
chmod +x run.sh

# Run the demo
./run.sh
```

## Manual Setup

If you prefer to set up the environment manually or are on a system where `run.sh` is not compatible:

1.  **Clone the repository** and navigate into this project directory:
    ```bash
    cd path/to/repo/microagent_helloworld
    ```
2.  **Install dependencies** using `uv`:
    ```bash
    uv sync
    ```
3.  **Set your LLM API key**:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
4.  **Run the demo**:
    ```bash
    uv run helloworld-demo
    ```
# MicroAgent HelloWorld Demo

A simple "Hello, World!" project that demonstrates how to use the `microagent` library.

The demo involves an agent that attempts to build the string "Hello, World!" by querying an LLM one character at a time. It showcases the core components of the library, including:
- A custom `MicroAgent` implementation.
- Using the `MicroAgentExecutor` to run the agent.
- Tracking execution statistics like cost and API calls.

## Prerequisites

1.  **Python & `uv`**: Ensure you have Python and the `uv` package manager installed.
2.  **LLM API Key**: An active API key is required. The demo automatically checks for the `OPENAI_API_KEY` environment variable.

## Quick Start (with `run.sh`)

The fastest way to run the demo is with the provided shell script, which handles all setup for you.

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Make the script executable (if you haven't already)
chmod +x run.sh

# Run the demo
./run.sh
```

## Manual Setup

If you prefer to set up the environment manually or are on a system where `run.sh` is not compatible:

1.  **Clone the repository** and navigate into this project directory:
    ```bash
    cd path/to/repo/microagent_helloworld
    ```
2.  **Install dependencies** using `uv`:
    ```bash
    uv sync
    ```
3.  **Set your LLM API key**:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
4.  **Run the demo**:
    ```bash
    uv run helloworld-demo
    ```
