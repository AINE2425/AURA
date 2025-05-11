from fastapi import FastAPI, HTTPException, Request
from keywords import abstract_to_keywords

app = FastAPI()


@app.post("/generate-keywords/")
async def generate_keywords(request: Request):
    try:
        body = await request.json()
        print("Received body:", body)  # Debugging line
        abstract = body.get("abstract")
        if not abstract:
            raise HTTPException(
                status_code=400, detail="Missing 'abstract' in request body"
            )

        keywords = abstract_to_keywords(abstract)
        return {"keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello World"}
