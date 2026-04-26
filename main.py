"""
Adaptive Document Processing — Pro Tier (Gemini 2.5 Pro)
Entry point
"""

import uvicorn
from config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.api.endpoints:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
