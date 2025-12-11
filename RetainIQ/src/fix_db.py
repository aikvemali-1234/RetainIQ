from sqlalchemy import create_engine, text

# Create engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./retainiq.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

print("Attempting to add created_at column...")

# Add the missing column
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE documents ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
        conn.commit()
        print("✅ SUCCESS! Added created_at column to documents table!")
    except Exception as e:
        error_msg = str(e)
        if "duplicate column" in error_msg.lower():
            print("✅ Column already exists - database is ready!")
        else:
            print(f"❌ Error: {e}")
            print("\n⚠️ The ALTER TABLE command didn't work.")
            print("Let's try the nuclear option instead...")

print("\nDone! Now restart your server with: uvicorn main:app --reload")