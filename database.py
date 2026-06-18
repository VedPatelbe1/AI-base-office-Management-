import numpy as np
from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String, LargeBinary,
    DateTime, Date, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

engine = create_engine("sqlite:///attendance.db", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    emp_code = Column(String, unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # 512-dim float32 face vector
    created_at = Column(DateTime, default=datetime.now)

    attendances = relationship("Attendance", back_populates="employee")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    date = Column(Date, default=date.today)
    check_in = Column(DateTime, default=datetime.now)   # check-in only

    employee = relationship("Employee", back_populates="attendances")


def embedding_to_bytes(emb):
    """NumPy array -> bytes for storing in the DB."""
    return np.asarray(emb, dtype=np.float32).tobytes()


def bytes_to_embedding(blob):
    """Bytes from the DB -> NumPy array for comparison."""
    return np.frombuffer(blob, dtype=np.float32)


def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized: attendance.db created with tables.")


if __name__ == "__main__":
    init_db()