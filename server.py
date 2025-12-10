from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app21 import ask_question  

app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask_agent(query: Query):
    try:
        response = ask_question(query.message)
        
        if not response:
            return JSONResponse(
                status_code=404,
                content={
                    "status": False,
                    "message": "No answer found",
                    "data": [{"answer": None}]
                }
            )
        
        if "maximum retries" in response.lower():
            return JSONResponse(
                status_code=429,
                content={
                    "status": False,
                    "message": "Maximum retry limit reached",
                    "data": [{"answer": None}]
                }
            )
        
        if "bad gateway" in response.lower():
            return JSONResponse(
                status_code=502,
                content={
                    "status": False,
                    "message": "Bad Gateway",
                    "data": [{"answer": None}]
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Data fetched successfully",
                "data": [{"answer": response}]
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": f"Internal Server Error: {str(e)}",
                "data": [{"answer": None}]
            }
        )


