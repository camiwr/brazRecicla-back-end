from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from classify_router import router as classify_router
from feedback_router import router as feedback_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

app.include_router(classify_router)
app.include_router(feedback_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
