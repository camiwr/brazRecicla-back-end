from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_db_connection

router = APIRouter()

class FeedbackCreate(BaseModel):
    user_name: str
    rating: int
    comment: str
    class_predicted: str

@router.post("/feedbacks/")
async def create_feedback(feedback: FeedbackCreate):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO feedbacks (user_name, rating, comment, class_predicted, created_at) VALUES (%s, %s, %s, %s, CURRENT_DATE)",
            (feedback.user_name, feedback.rating, feedback.comment, feedback.class_predicted)
        )
        conn.commit()
        return {"message": "Feedback enviado com sucesso!"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Erro ao inserir feedback: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@router.get("/feedbacks/")
async def get_feedbacks():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM feedbacks ORDER BY created_at DESC")
        feedbacks = cursor.fetchall()
        return feedbacks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar feedbacks: {str(e)}")
    finally:
        cursor.close()
        conn.close()
