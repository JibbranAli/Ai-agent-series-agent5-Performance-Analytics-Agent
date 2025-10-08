# 🚀 Deployment Guide

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai_agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp env.example .env
# Edit .env with your GEMINI_API_KEY
```

### 4. Run Application
```bash
streamlit run main.py
```

## Production Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables
4. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: Model to use (default: gemini-2.0-flash)
- `DEBUG`: Enable debug mode (default: false)

## Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src
```

## Monitoring

- Logs are stored in `logs/` directory
- Application metrics available via Streamlit
- Error tracking through Python logging

## Security

- API keys stored in environment variables
- No sensitive data in codebase
- Input validation for all user data
- Secure file upload handling
