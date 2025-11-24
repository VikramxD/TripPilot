# TripPilot

**AI-Powered Travel Companion with Multi-Agent Architecture**

TripPilot is a 2025-ready intelligent travel planning system that combines multiple specialized AI agents to create personalized itineraries, research destinations, estimate budgets, and provide local insights.

## Features

- **Multi-Agent Architecture** - Specialized AI agents for research, itinerary planning, budgeting, and local expertise
- **RAG Pipeline** - Retrieval-Augmented Generation with ChromaDB for contextual travel knowledge
- **Real-Time Search** - Live web search for up-to-date travel information
- **Weather Integration** - Automatic weather forecasts and packing suggestions
- **Structured Outputs** - Type-safe responses using Pydantic v2
- **Async FastAPI** - High-performance REST API with async endpoints
- **CLI Interface** - Rich command-line interface for terminal users
- **Observability** - Structured logging with optional Weights & Biases integration

## Architecture

```
                     TripPilot Orchestrator
                   (Coordinates all agents)

        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Research    │      │   Itinerary   │      │    Budget     │
│    Agent      │      │    Agent      │      │    Agent      │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Local Expert  │      │   Cultural    │      │ Deal Finder   │
│    Agent      │      │   Advisor     │      │    Agent      │
└───────────────┘      └───────────────┘      └───────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │   RAG Pipeline    │
                    │  (ChromaDB/Lance) │
                    └───────────────────┘
```

## Agents

| Agent | Description |
|-------|-------------|
| **Research Agent** | Gathers comprehensive destination information, attractions, and practical tips |
| **Itinerary Agent** | Creates personalized day-by-day travel plans based on preferences |
| **Budget Agent** | Estimates costs, creates breakdowns, and provides money-saving tips |
| **Local Expert Agent** | Provides insider knowledge, hidden gems, and authentic experiences |
| **Cultural Advisor** | Offers cultural context, etiquette, and customs information |
| **Deal Finder Agent** | Finds travel deals, discounts, and booking strategies |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VikramxD/TripPilot.git
cd TripPilot

# Install with pip
pip install -e ".[dev]"

# Or with uv (recommended)
uv pip install -e ".[dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### CLI Usage

```bash
# Plan a complete trip
trippilot plan "Tokyo, Japan" --days 7 --budget moderate --style cultural

# Quick destination research
trippilot research "Barcelona, Spain"

# Estimate budget
trippilot budget "Paris, France" --days 5 --level luxury

# Start API server
trippilot serve --port 8000
```

### API Usage

```bash
# Start the server
trippilot serve

# Plan a trip via API
curl -X POST http://localhost:8000/api/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Tokyo, Japan",
    "duration_days": 7,
    "travelers": 2,
    "budget_level": "moderate",
    "travel_styles": ["cultural", "foodie"]
  }'

# Quick research
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"destination": "Barcelona, Spain"}'

# Get local tips
curl -X POST http://localhost:8000/api/v1/local-tips \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Rome, Italy",
    "interests": ["food", "history"]
  }'
```

### Python SDK

```python
import asyncio
from trippilot import TripPilotOrchestrator
from trippilot.schemas import TravelQuery, TravelPreferences, TravelStyle, BudgetLevel

async def plan_my_trip():
    # Create orchestrator
    orchestrator = TripPilotOrchestrator()

    # Define your trip
    query = TravelQuery(
        destination="Kyoto, Japan",
        duration_days=5,
        travelers=2,
        preferences=TravelPreferences(
            styles=[TravelStyle.CULTURAL, TravelStyle.RELAXATION],
            budget_level=BudgetLevel.MODERATE,
            interests=["temples", "gardens", "tea ceremony"],
        ),
    )

    # Get recommendation
    result = await orchestrator.plan_trip(query)

    if result.success:
        itinerary = result.recommendation.itinerary
        print(f"Trip: {itinerary.title}")
        print(f"Budget: ${itinerary.budget.total_estimated:,.0f}")

        for day in itinerary.daily_plans:
            print(f"\nDay {day.day_number}: {day.title}")
            for activity in day.morning + day.afternoon:
                print(f"  - {activity.name}")

asyncio.run(plan_my_trip())
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# With external ChromaDB
docker-compose --profile with-chromadb up -d

# View logs
docker-compose logs -f trippilot
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/plan` | POST | Plan a complete trip |
| `/api/v1/research` | POST | Research a destination |
| `/api/v1/local-tips` | POST | Get local tips and hidden gems |
| `/api/v1/budget` | POST | Estimate trip budget |
| `/api/v1/destinations/{name}` | GET | Quick destination info |

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/anthropic) | `openai` |
| `DEFAULT_MODEL` | Model for complex tasks | `gpt-4o` |
| `FAST_MODEL` | Model for simple tasks | `gpt-4o-mini` |
| `VECTOR_DB` | Vector database (chromadb/lancedb) | `chromadb` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `MAX_SEARCH_RESULTS` | Web search results limit | `10` |
| `WANDB_ENABLED` | Enable W&B logging | `false` |

## Project Structure

```
TripPilot/
├── src/trippilot/
│   ├── agents/           # AI agents
│   │   ├── base.py       # Base agent class
│   │   ├── research.py   # Research agent
│   │   ├── itinerary.py  # Itinerary planner
│   │   ├── budget.py     # Budget agent
│   │   └── local_expert.py # Local expert
│   ├── api/              # FastAPI application
│   │   └── app.py        # API endpoints
│   ├── core/             # Core components
│   │   ├── config.py     # Configuration
│   │   └── orchestrator.py # Agent orchestration
│   ├── rag/              # RAG pipeline
│   │   ├── embeddings.py # Embedding service
│   │   ├── vector_store.py # Vector database
│   │   └── retriever.py  # Knowledge retriever
│   ├── schemas/          # Pydantic models
│   │   └── travel.py     # Travel schemas
│   ├── tools/            # Agent tools
│   │   ├── search.py     # Web search
│   │   └── weather.py    # Weather API
│   ├── utils/            # Utilities
│   │   └── logging.py    # Logging setup
│   └── cli.py            # CLI interface
├── tests/                # Test suite
├── data/                 # Data storage
├── scripts/              # Legacy scripts
├── pyproject.toml        # Project config
├── Dockerfile            # Container build
├── docker-compose.yml    # Container orchestration
└── .env.example          # Environment template
```

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **LLM** | OpenAI GPT-4o, Anthropic Claude |
| **Vector DB** | ChromaDB, LanceDB |
| **Embeddings** | Sentence Transformers |
| **Web Framework** | FastAPI |
| **CLI** | Typer + Rich |
| **Validation** | Pydantic v2 |
| **HTTP Client** | httpx, aiohttp |
| **Search** | DuckDuckGo |
| **Observability** | structlog, W&B |
| **Testing** | pytest, pytest-asyncio |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/trippilot

# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/trippilot
```

## Roadmap

- [ ] Multi-destination trip planning
- [ ] Flight and hotel booking integration
- [ ] Mobile app with React Native
- [ ] Voice interface with Whisper
- [ ] Real-time price tracking
- [ ] Social features for trip sharing
- [ ] Offline mode with local LLMs

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with AI by [VikramxD](https://github.com/VikramxD)**
