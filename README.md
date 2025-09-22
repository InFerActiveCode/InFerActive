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


## Backend (server) Configuration

### Command Line Options

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


## Backend Architecture

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

## Contact

For questions or support, please contact: inferactive@proton.me