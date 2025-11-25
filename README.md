# TripPilot

An intelligent travel companion leveraging cutting-edge AI to create personalized itineraries and recommendations.

## Features

**Core Capabilities:**
- **Personalized Itineraries**: AI-powered travel plans tailored to your preferences
- **Real-time Recommendations**: Up-to-date suggestions for attractions and activities
- **Natural Language Interface**: Interact using conversational language
- **Retrieval Augmented Generation (RAG)**: Enhance responses with relevant knowledge
- **Scalable Architecture**: Built to handle varying loads efficiently

**Agentic UI Features:**
- **Live Agent Reasoning**: Watch the AI think and plan in real-time
- **Tool Visualization**: See which tools the agent uses and their results
- **Interactive Chat**: Natural conversation interface for travel planning
- **Rich Results**: Beautiful cards for destinations, hotels, restaurants, and itineraries

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/VikramxD/TripPilot.git
cd TripPilot

# Install dependencies
pip install -r requirements.txt

# Run the Agentic UI
python app.py
```

The app will open in your browser at `http://localhost:7860`

### Command Line Options

```bash
python app.py --port 8080        # Custom port
python app.py --share            # Create public link
python app.py --debug            # Enable debug mode
python app.py --no-browser       # Don't open browser
```

## Agentic UI

The Agentic UI provides a unique window into AI reasoning:

### Agent Tools
The travel agent has access to these tools:
- `search_destinations` - Find destinations matching your preferences
- `find_hotels` - Search for accommodations
- `get_restaurants` - Restaurant recommendations
- `create_itinerary` - Generate day-by-day plans
- `get_attractions` - Top attractions and activities
- `check_weather` - Weather forecasts
- `get_travel_tips` - Local tips and insights
- `estimate_budget` - Trip cost estimation

### Example Queries
Try these to get started:
- "Plan a 5-day trip to Paris with focus on art and cuisine"
- "Find budget-friendly hotels in Tokyo"
- "What are the top attractions in Barcelona?"
- "Create a romantic itinerary for Bali"
- "Estimate the budget for a week in Rome"

## Project Structure

```
TripPilot/
├── app.py                    # Main entry point
├── tripilot/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── agents/
│   │   ├── __init__.py
│   │   └── travel_agent.py   # AI travel agent with tools
│   ├── ui/
│   │   ├── __init__.py
│   │   └── agentic_ui.py     # Gradio-based UI
│   └── data/
│       ├── __init__.py
│       └── destinations.py   # Sample destination data
├── scripts/
│   └── dataset.py            # Data processing utilities
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **UI Framework** | Gradio |
| **Backend** | FastAPI |
| **ML Model** | PyTorch Lightning |
| **Vector Database** | LanceDB |
| **Inference** | LitServe |
| **Cloud Infrastructure** | AWS (EC2, S3) |
| **Monitoring** | Weights and Biases |

## Architecture

Our system utilizes a Retrieval Augmented Generation (RAG) approach:

1. **Document Ingestion**: Preprocess documents from knowledge base
2. **Embedding Generation**: Create embeddings for documents and queries
3. **Vector Storage**: Store embeddings in LanceDB for efficient retrieval
4. **User Interaction**: Process queries through the Agentic UI
5. **Contextual Retrieval**: Fetch relevant information from LanceDB
6. **Response Generation**: Create personalized travel recommendations

## Development

### Running in Development Mode

```bash
python app.py --debug
```

### Project Dependencies

Core dependencies:
- `gradio` - Interactive UI
- `pydantic-settings` - Configuration management
- `transformers` - NLP models
- `torch` - Deep learning framework
- `lightning` - Training framework
- `fastapi` - API backend

## Monitoring

We use Weights and Biases for comprehensive monitoring:
- Model accuracy and perplexity
- API response time
- Inference latency

## Acknowledgements

- Gradio team for the amazing UI framework
- Lightning team for the Lightning framework
- LanceDB for efficient vector storage

---

Made with passion by VikramxD
