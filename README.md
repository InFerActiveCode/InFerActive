# InferActive

**Interactive token-level inference server for language models with real-time tree exploration and advanced sampling methods.**

InferActive enables real-time, token-by-token generation with full probability tracking, interactive tree exploration, and sophisticated sampling techniques like Sequential Monte Carlo (SMC). Perfect for research, interactive applications, and understanding model behavior at the token level.

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

### Frontend
```bash 
cd frontend
npm start
```

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

**Debugging:**
- `--log-level` - Logging verbosity: `debug`, `info`, `warning`, `error`

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

**Model Requirements:**
- HuggingFace format with `config.json`
- Compatible tokenizer files
- Model weights in `.safetensors` or `.bin` format

#### Generate with SMC
```json
{
    "type": "generate_with_smc",
    "request_id": "gen-001",
    "input_text": "Your prompt here",
    "k": 5,
    "particlenum": 20,
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "min_p": 0.05
}
```

#### Explore Tree Node
```json
{
    "type": "explore_node", 
    "request_id": "gen-001",
    "node_id": "node-to-explore",
    "k": 5,
    "temperature": 0.7
}
```

### Response Types

#### Model Status
```json
{
    "type": "model_status",
    "request_id": "load-001",
    "status": "loaded",
    "message": "Model loaded successfully"
}
```

#### Generation Result
```json
{
    "type": "tree_result",
    "request_id": "gen-001", 
    "tree": {
        "id": "root",
        "text": "Your prompt",
        "children": [
            {
                "id": "child-1",
                "token_id": 123,
                "text": "token",
                "prob": 0.85,
                "score": 0.85,
                "children": [...]
            }
        ]
    }
}
```

#### Real-Time Updates
```json
{
    "type": "update",
    "tree": { ... }
}
```

### HTTP Endpoints

- **GET /** - Server information and status
- **GET /health** - Health check for monitoring
- **GET /stats** - Detailed server statistics

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
```

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
