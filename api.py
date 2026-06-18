from datetime import date, datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, Employee, Attendance

app = FastAPI(title="Attendance API")

# Allow the Streamlit dashboard (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def health():
    return {"status": "ok", "service": "attendance-api"}


@app.get("/employees")
def list_employees(db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "emp_code": e.emp_code,
            "created_at": e.created_at,
        }
        for e in employees
    ]


@app.get("/attendance")
def get_attendance(day: str = None, db: Session = Depends(get_db)):
    # Default to today if no date is passed
    if day:
        try:
            target = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    else:
        target = date.today()

    records = (
        db.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .filter(Attendance.date == target)
        .order_by(Attendance.check_in)
        .all()
    )

    return {
        "date": target.isoformat(),
        "count": len(records),
        "records": [
            {
                "name": emp.name,
                "emp_code": emp.emp_code,
                "check_in": att.check_in.strftime("%H:%M:%S") if att.check_in else None,
            }
            for att, emp in records
        ],
    }


@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int, db: Session = Depends(get_db)):
    emp = db.query(Employee).filter_by(id=emp_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    db.query(Attendance).filter_by(employee_id=emp_id).delete()  # remove their records too
    db.delete(emp)
    db.commit()
    return {"deleted": emp_id}