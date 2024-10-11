import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_db_connection():
    conn = psycopg2.connect(
        host="aws-0-sa-east-1.pooler.supabase.com",
        database="postgres",  
        user="postgres.zcbswpxewpxprtffxhza", 
        port="6543",    
        password="brazrecicla022", 
        cursor_factory=RealDictCursor
    )
    return conn
