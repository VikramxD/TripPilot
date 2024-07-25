# 🌍 AI Itinerary and Travel Planner

An intelligent travel companion leveraging cutting-edge AI to create personalized itineraries and recommendations.




## 🚀 Features
• Personalized Itineraries: AI-powered travel plans tailored to your preferences
• Real-time Recommendations: Up-to-date suggestions for attractions and activities
• Natural Language Interface: Interact using conversational language
• Retrieval Augmented Generation (RAG): Enhance responses with relevant knowledge
• Scalable Architecture: Built to handle varying loads efficiently

## 🛠️ Tech Stack
• Backend: FastAPI
• ML Model: PyTorch Lightning
• Vector Database: LanceDB
• Inference: Triton Inference Server
• Cloud Infrastructure: AWS (EC2, S3)
• Monitoring: Weights and Biases
• CI/CD: GitHub Actions

## 🏗️ Architecture

![image](https://github.com/user-attachments/assets/0c1bd48c-ff8d-40e4-8bb0-6b219d61a03e)


Our system utilizes a Retrieval Augmented Generation (RAG) approach:
1. Document Ingestion: Preprocess documents from Enterprise Knowledge Base
2. Embedding Generation: Create embeddings for documents and queries
3. Vector Storage: Store embeddings in LanceDB for efficient retrieval
4. User Interaction: Process queries through FastAPI
5. Contextual Retrieval: Fetch relevant information from LanceDB
6. Response Generation: Create personalized travel recommendations using PyTorch Lightning model

## 🚦 Getting Started
Prerequisites:
• Python 3.10+
• AWS EC2
• Docker


## 📊 Monitoring
We use Weights and Biases for comprehensive monitoring. Key metrics include:
• Model accuracy and perplexity
• API response times
• Inference latency
• Resource utilization (CPU, GPU, Memory)
Access the dashboard at [wandb.ai/your-project](https://wandb.ai/your-project)

## 🔧 Deployment
The system is deployed on AWS EC2 instances:
• FastAPI server for handling user requests
• PyTorch Lightning model for generating responses
• LanceDB for efficient vector storage and retrieval
• Triton Inference Server for optimized model serving

GitHub Actions automate the deployment process:
• Runs tests on pull requests
• Deploys to staging environment for review
• Deploys to production on merges to main branch

## 🙏 Acknowledgements
• Lightning team for the Lightning framework
• LanceDB for efficient vector storage
• NVIDIA for Triton Inference Server


Made with ❤️ by VikramXD
