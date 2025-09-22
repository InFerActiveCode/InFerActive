# InferActive

<img width="4480" height="2056" alt="Image" src="https://github.com/user-attachments/assets/a020068e-e403-43f4-a137-6b8c737c85b9" />

InFerActive, a novel interactive system that supports optimized visualization and direct manipulation of LLMs for human evaluation of LLMs.

## Features

🚀 **Token-Level Inference**: Real-time generation with complete probability tracking for every token  
🌳 **Interactive Tree Exploration**: Expand and explore generation trees interactively  
🎯 **Sequential Monte Carlo**: Advanced particle-based sampling for higher quality generation  
⚡ **Batched Processing**: Efficient automatic batching of inference requests  
🔄 **Real-Time Communication**: WebSocket API for instant updates and interaction  
🤖 **Multi-Model Support**: Llama 3.x, Qwen, and EXAONE model families  
🛡️ **Smart Model Management**: Automatic loading/unloading with GPU memory optimization  
⚙️ **Highly Configurable**: Flexible parameters for different use cases and hardware

## Quick Start

### Try Demo
https://inferactivedemo.netlify.app

### Installation

```bash
# Clone the repository
git clone https://github.com/inferactivecode/inferactive.git
cd inferactive

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Start server with your model
inferactive-server --model-path /path/to/your/model --model-type llama

# Advanced configuration
inferactive-server \
  --model-path /path/to/qwen-model \
  --model-type qwen \
  --port 8008 \
  --gpu-id 1 \
  --batch-size 32 \
  --log-level debug
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at `http://localhost:3000` and automatically connects to the backend server at `http://localhost:8008`.

## Advanced Features

### Sequential Monte Carlo (SMC) Generation

SMC maintains multiple generation candidates (particles) and uses intelligent resampling to focus computational resources on the most promising paths:

```python
{
    "type": "generate_with_smc",
    "input_text": "Explain quantum computing",
    "k": 8,                    # Top-k candidates per step
    "particlenum": 25,         # Number of particles
    "max_tokens": 100,         # Maximum generation length
    "temperature": 0.8,        # Sampling temperature
    "top_p": 0.9,             # Nucleus sampling
    "min_p": 0.02             # Minimum probability threshold
}
```

### Interactive Tree Exploration

Explore specific nodes in the generation tree to understand model behavior:

```python
# First generate initial tree, then explore specific branches
{
    "type": "explore_node",
    "request_id": "gen-001",      # Reference to existing tree
    "node_id": "target-node-id",  # Specific node to expand
    "k": 5,                       # Alternatives per expansion
    "temperature": 0.7
}
```


## Configuration

### Command Line Options

```bash
inferactive-server --help
```

**Essential Options:**
- `--model-path` - Path to model directory (required)
- `--model-type` - Model family: `llama`, `qwen`, `exaone`
- `--host` - Server host (default: `0.0.0.0`)
- `--port` - Server port (default: `8008`)
- `--gpu-id` - GPU device to use (default: `0`)

**Performance Tuning:**
- `--batch-size` - Maximum batch size for efficiency (default: `16`)
- `--batch-timeout` - Batch collection timeout in seconds (default: `0.1`)
- `--model-unload-timeout` - Auto-unload after inactivity (default: `600`)

### Environment Configuration

Create a `.env` file for persistent configuration:

```bash
INFERACTIVE_MODEL_PATH=/path/to/your/model
INFERACTIVE_MODEL_TYPE=llama
INFERACTIVE_GPU_ID=0
INFERACTIVE_PORT=8008
INFERACTIVE_BATCH_SIZE=16
INFERACTIVE_LOG_LEVEL=INFO
```

Then simply run: `inferactive-server`

### Model Support

**Supported Architectures:**
- **Llama 3.x**: All Llama 3.1 and 3.2 variants
- **Qwen**: Qwen family models with thinking capabilities
- **EXAONE**: EXAONE 3.5 series models


## Architecture

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   └── cli.py               # Command line interface
├── config/
│   └── settings.py          # Configuration management
├── model/
│   ├── token_node.py        # Tree node data structure
│   └── manager.py           # Model lifecycle management
├── inference/
│   └── engine.py            # Batched inference engine
├── api/
│   ├── connections.py       # WebSocket & session management
│   └── handlers.py          # Message routing & handling
└── utils/
    └── logging.py           # Logging configuration
    
## Performance Optimization

### GPU Memory Management
- Models automatically unload after configurable inactivity period
- Intelligent batch processing reduces GPU memory fragmentation
- Automatic cleanup and restart on critical GPU errors

### Batch Processing Optimization
- **Higher Throughput**: Increase `--batch-size` for more parallel requests
- **Lower Latency**: Decrease `--batch-timeout` for faster response times
- **Memory vs Speed**: Balance based on your GPU capacity

## Development

### Setup Development Environment
```bash
# Install in development mode
pip install -e ".[dev]"

```


## License

MIT License
