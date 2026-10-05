from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.hybrid import MovieGraphEngine

app = FastAPI(title="Movie Similarity Graph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = MovieGraphEngine()

@app.get("/movies")
def list_movies(request: Request):
    return engine.movies_df[['movieId', 'title', 'genres']].to_dict(orient="records")

@app.get("/graph/{movie_id}")
def get_movie_graph(movie_id: int, request: Request, threshold: float = Query(0.15, ge=0.0, le=1.0)):
    result = engine.get_related_movies(movie_id, threshold=threshold)
    if not result or not result["connections"]:
        raise HTTPException(status_code=404, detail="Unknown or isolated movie with no strong connections.")
    return result