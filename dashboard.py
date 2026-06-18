import os
import io
import sys
import json
import html as _html
import subprocess
import threading
from datetime import date, datetime, time, timedelta

import cv2
import av
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from config import RTSP_URL
from database import SessionLocal, Employee, Attendance, embedding_to_bytes

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

LATE_AFTER = time(11, 15)   # check-ins after this time show as "Late"


def load_env(filepath=".env"):
    if not os.path.exists(filepath):
        return
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

HR_SYSTEM_PROMPT = """You are a helpful HR Assistant for a company. You help employees and HR staff with:
- Leave & attendance policies
- Payroll and salary queries
- Onboarding and offboarding processes
- Employee benefits and perks
- Performance review processes
- Workplace policies and code of conduct
- Recruitment and job postings
- Training and development programs
- Grievance handling

Be professional, empathetic, and concise. If you don't know a specific company policy,
ask the user to check with their HR department directly."""


st.set_page_config(page_title="FaceTrack Attendance", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #FFFFFF; }
[data-testid="stHeader"] { background: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #F5F5F7; border-right: 1px solid #ececef; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stMarkdown, h1, h2, h3, h4 { color: #1A1A1A; }
[data-testid="stSidebar"] * { color: #1A1A1A; }
small, [data-testid="stCaptionContainer"] { color: #6b6b6b !important; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }

.stTextInput input, .stNumberInput input, .stDateInput input {
  background: #FFFFFF; color: #1A1A1A; border: 1px solid #d6d6db;
}
.stButton button, .stDownloadButton button {
  background: #FFFFFF; color: #1A1A1A; border: 1px solid #d6d6db;
}
.stButton button:hover, .stDownloadButton button:hover {
  border-color: #FF4B4B; color: #FF4B4B;
}
.stButton button[kind="primary"] { background: #FF4B4B; color: #FFFFFF; border: none; }

.mcard { background:#FFFFFF; border:1px solid #ececef; border-radius:12px; padding:16px 18px; }
.mlab { font-size:13px; color:#6b6b6b; margin-bottom:6px; }
.mval { font-size:26px; font-weight:600; color:#1A1A1A; line-height:1.1; }

.lt table { width:100%; border-collapse:collapse; font-size:14px; }
.lt th { text-align:left; background:#F5F5F7; color:#555; padding:9px 12px;
         border-bottom:1px solid #e6e6ea; font-weight:600; }
.lt td { padding:9px 12px; color:#1A1A1A; border-bottom:1px solid #efeff2; }
.pill-present { background:#E1F5EE; color:#0F6E56; padding:2px 10px; border-radius:8px; font-size:12px; }
.pill-late { background:#FAEEDA; color:#854F0B; padding:2px 10px; border-radius:8px; font-size:12px; }

.uwrap { display:flex; justify-content:flex-end; margin:10px 0; }
.ubub { background:#F0F0F3; color:#1A1A1A; padding:10px 16px;
        border-radius:20px 20px 4px 20px; max-width:75%; font-size:15px; line-height:1.5; }
.greet { text-align:center; margin:70px 0 28px; }
.greet h2 { font-size:30px; font-weight:600; color:#1A1A1A; margin:0; }
.greet p { color:#6b6b6b; margin:8px 0 0; font-size:15px; }

[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background:#FFFFFF !important; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
  background:#FFFFFF !important; border-radius:26px;
}
[data-testid="stChatInput"] { border:1px solid #d6d6db; }
[data-testid="stChatInput"] textarea {
  background:transparent !important; color:#1A1A1A !important;
  -webkit-text-fill-color:#1A1A1A !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color:#9aa0a6 !important; -webkit-text-fill-color:#9aa0a6 !important;
}
[data-testid="stChatInputSubmitButton"] { color:#1A1A1A !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(480, 480))
    return app


model = load_model()


def status_badge(check_in):
    if check_in and check_in.time() > LATE_AFTER:
        return "Late"
    return "Present"


def metric_card(col, label, value):
    col.markdown(
        f"<div class='mcard'><div class='mlab'>{label}</div>"
        f"<div class='mval'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def render_table(df):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            if c == "Status":
                cls = "pill-late" if v == "Late" else "pill-present"
                v = f"<span class='{cls}'>{v}</span>"
            cells += f"<td>{v}</td>"
        rows += f"<tr>{cells}</tr>"
    st.markdown(
        f"<div class='lt'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def build_hr_context():
    session = SessionLocal()
    employees = session.query(Employee).all()
    today = date.today()
    since = today - timedelta(days=30)
    records = (
        session.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .filter(Attendance.date >= since)
        .order_by(Attendance.date, Attendance.check_in)
        .all()
    )
    session.close()

    emp_lines = [f"- {e.name} ({e.emp_code})" for e in employees]
    att_lines = [
        f"{att.date.strftime('%Y-%m-%d')} | {emp.name} ({emp.emp_code}) | "
        f"check-in {att.check_in.strftime('%H:%M:%S') if att.check_in else '-'} | "
        f"{status_badge(att.check_in)}"
        for att, emp in records
    ]
    return (
        f"Today's date is {today.strftime('%Y-%m-%d')}. "
        f"On-time cutoff is {LATE_AFTER.strftime('%H:%M')} (after that = Late).\n\n"
        f"EMPLOYEES ({len(employees)}):\n" + ("\n".join(emp_lines) or "None") + "\n\n"
        f"ATTENDANCE (last 30 days):\n" + ("\n".join(att_lines) or "No records") + "\n\n"
        "Answer using ONLY this company data plus general HR knowledge."
    )


# ---------------- HR assistant action tools ----------------
def tool_query_attendance(date_str=None):
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
    except Exception:
        return "Invalid date. Use YYYY-MM-DD."
    session = SessionLocal()
    records = (
        session.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .filter(Attendance.date == target).order_by(Attendance.check_in).all()
    )
    session.close()
    if not records:
        return f"No check-ins on {target}."
    lines = [
        f"{emp.name} ({emp.emp_code}) | "
        f"{att.check_in.strftime('%H:%M:%S') if att.check_in else '-'} | {status_badge(att.check_in)}"
        for att, emp in records
    ]
    return f"Attendance for {target} ({len(records)}):\n" + "\n".join(lines)


def tool_start_attendance():
    subprocess.Popen([sys.executable, "attendance.py"])
    return "Opened the live attendance camera window (press q there to stop)."


def tool_open_enrollment(name, emp_code):
    if not name or not emp_code:
        return "Need both a name and an employee code to enroll."
    session = SessionLocal()
    existing = session.query(Employee).filter_by(emp_code=emp_code).first()
    session.close()
    if existing:
        return f"Employee code {emp_code} already exists ({existing.name})."
    subprocess.Popen([sys.executable, "enroll.py", name, emp_code])
    return (f"Opened the enrollment window for {name} ({emp_code}). Stand at the camera, "
            "press SPACE to capture several angles, then q to save.")


def tool_delete_employee(emp_code, confirmed=False):
    if not confirmed:
        return f"CONFIRMATION REQUIRED: ask the user to confirm deleting employee {emp_code} before proceeding."
    session = SessionLocal()
    emp = session.query(Employee).filter_by(emp_code=emp_code).first()
    if not emp:
        session.close()
        return f"No employee found with code {emp_code}."
    name = emp.name
    session.query(Attendance).filter_by(employee_id=emp.id).delete()
    session.delete(emp)
    session.commit()
    session.close()
    return f"Deleted {name} ({emp_code}) and all their attendance records."


def tool_delete_attendance(date_str, confirmed=False):
    if not confirmed:
        return f"CONFIRMATION REQUIRED: ask the user to confirm deleting attendance for {date_str}."
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return "Invalid date. Use YYYY-MM-DD."
    session = SessionLocal()
    n = session.query(Attendance).filter_by(date=target).delete()
    session.commit()
    session.close()
    return f"Deleted {n} attendance record(s) for {target}."


HR_TOOLS = [
    {"type": "function", "function": {
        "name": "query_attendance",
        "description": "Get attendance check-ins for a date (YYYY-MM-DD). Omit date for today.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "Date in YYYY-MM-DD"}}}}},
    {"type": "function", "function": {
        "name": "start_attendance",
        "description": "Open the live face-recognition camera window to take attendance.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "open_enrollment",
        "description": "Open the camera enrollment window to register a new employee. The person must be at the camera.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "emp_code": {"type": "string"}},
            "required": ["name", "emp_code"]}}},
    {"type": "function", "function": {
        "name": "delete_employee",
        "description": "Delete a specific employee by code and all their attendance. Destructive; confirm first.",
        "parameters": {"type": "object", "properties": {
            "emp_code": {"type": "string"},
            "confirmed": {"type": "boolean", "description": "true only after the user explicitly confirms"}},
            "required": ["emp_code"]}}},
    {"type": "function", "function": {
        "name": "delete_attendance",
        "description": "Delete all attendance for a date (YYYY-MM-DD). Destructive; confirm first.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string"},
            "confirmed": {"type": "boolean"}},
            "required": ["date"]}}},
]


def run_hr_tool(name, args):
    try:
        if name == "query_attendance":
            return tool_query_attendance(args.get("date"))
        if name == "start_attendance":
            return tool_start_attendance()
        if name == "open_enrollment":
            return tool_open_enrollment(args.get("name", ""), args.get("emp_code", ""))
        if name == "delete_employee":
            return tool_delete_employee(args.get("emp_code", ""), args.get("confirmed", False))
        if name == "delete_attendance":
            return tool_delete_attendance(args.get("date", ""), args.get("confirmed", False))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def hr_answer(api_key, conversation, user_input, data_context):
    try:
        client = OpenAI(api_key=api_key)
        system = (
            HR_SYSTEM_PROMPT + "\n\n--- COMPANY DATA ---\n" + data_context +
            "\n\n--- ACTIONS ---\nYou can use tools to: query attendance, start the attendance "
            "camera, open the enrollment window for a new hire (collect their full name and an "
            "employee code first), delete a specific employee by code, or delete a day's attendance. "
            "For ANY delete action, first ask the user to confirm and only set confirmed=true after "
            "they clearly say yes."
        )
        messages = [{"role": "system", "content": system}] + conversation + \
                   [{"role": "user", "content": user_input}]

        for _ in range(5):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=HR_TOOLS,
                temperature=0.3,
                max_tokens=800,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content.strip() if msg.content else "(no response)"
            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result = run_hr_tool(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "Stopped after several tool steps. Please try again."
    except Exception as e:
        return f"Error: {e}"


class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_embedding = None
        self.face_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = model.get(img)
        with self.lock:
            self.face_count = len(faces)
            self.latest_embedding = faces[0].normed_embedding if len(faces) == 1 else None
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ====================== SIDEBAR NAV ======================
with st.sidebar:
    st.markdown("## FaceTrack")
    st.caption("Face Recognition Attendance")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Dashboard", "Register Employee", "Take Attendance", "Reports", "HR Assistant"],
        label_visibility="collapsed",
    )


# ====================== DASHBOARD ======================
if page == "Dashboard":
    st.title("Attendance Dashboard")
    st.caption(datetime.now().strftime("%A, %d %B %Y"))

    session = SessionLocal()
    total = session.query(Employee).count()
    today = date.today()
    records = (
        session.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .filter(Attendance.date == today)
        .order_by(Attendance.check_in)
        .all()
    )
    session.close()

    present = len(records)
    absent = max(total - present, 0)
    rate = round(present / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total employees", total)
    metric_card(c2, "Present today", present)
    metric_card(c3, "Absent", absent)
    metric_card(c4, "Attendance rate", f"{rate}%")

    st.write("")
    st.subheader("Today's check-ins")
    if records:
        df = pd.DataFrame([
            {
                "Code": emp.emp_code,
                "Name": emp.name,
                "Check-in": att.check_in.strftime("%H:%M:%S") if att.check_in else "",
                "Status": status_badge(att.check_in),
            }
            for att, emp in records
        ])
        render_table(df)
    else:
        st.info("No check-ins yet today. Open 'Take Attendance' to start.")

    st.write("")
    if st.button("Refresh"):
        st.rerun()


# ====================== REGISTER EMPLOYEE ======================
elif page == "Register Employee":
    st.title("Register Employee")

    if "enroll_embeddings" not in st.session_state:
        st.session_state.enroll_embeddings = []

    col_form, col_list = st.columns([2, 1])

    with col_form:
        name = st.text_input("Employee name")
        emp_code = st.text_input("Employee code (e.g. EMP001)")
        st.caption("Allow camera access. A green box tracks your face. "
                   "Click 'Capture sample' from several angles (min 3), then 'Save Employee'.")

        ctx = webrtc_streamer(
            key="register-cam",
            video_processor_factory=FaceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

        cap_col, clr_col = st.columns(2)
        if cap_col.button("Capture sample", width="stretch"):
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    emb = ctx.video_processor.latest_embedding
                if emb is not None:
                    st.session_state.enroll_embeddings.append(emb)
                    st.success(f"Captured. Total: {len(st.session_state.enroll_embeddings)}")
                else:
                    st.warning("Need exactly one face clearly in view.")
            else:
                st.warning("Start the camera first.")

        if clr_col.button("Clear samples", width="stretch"):
            st.session_state.enroll_embeddings = []
            st.info("Cleared.")

        st.markdown(f"**Samples captured:** {len(st.session_state.enroll_embeddings)}")

        if st.button("Save Employee", type="primary"):
            if not name.strip() or not emp_code.strip():
                st.error("Enter both name and employee code.")
            elif len(st.session_state.enroll_embeddings) < 3:
                st.error("Capture at least 3 samples.")
            else:
                session = SessionLocal()
                if session.query(Employee).filter_by(emp_code=emp_code.strip()).first():
                    st.error(f"Code '{emp_code}' already exists.")
                    session.close()
                else:
                    avg = np.mean(st.session_state.enroll_embeddings, axis=0)
                    avg = avg / np.linalg.norm(avg)
                    session.add(Employee(
                        name=name.strip(),
                        emp_code=emp_code.strip(),
                        embedding=embedding_to_bytes(avg),
                    ))
                    session.commit()
                    session.close()
                    st.session_state.enroll_embeddings = []
                    st.success(f"Saved '{name}' ({emp_code}).")

    with col_list:
        st.subheader("Enrolled")
        session = SessionLocal()
        employees = session.query(Employee).all()
        if employees:
            df = pd.DataFrame([{"ID": e.id, "Name": e.name, "Code": e.emp_code} for e in employees])
            render_table(df)
        else:
            st.info("None yet.")
        session.close()

        del_id = st.number_input("Delete by ID", min_value=0, step=1, value=0)
        if st.button("Delete employee"):
            if del_id > 0:
                session = SessionLocal()
                emp = session.query(Employee).filter_by(id=int(del_id)).first()
                if emp:
                    session.query(Attendance).filter_by(employee_id=int(del_id)).delete()
                    session.delete(emp)
                    session.commit()
                    st.success(f"Deleted employee {del_id}.")
                else:
                    st.error("Employee not found.")
                session.close()


# ====================== TAKE ATTENDANCE ======================
elif page == "Take Attendance":
    st.title("Take Attendance")
    st.write(
        "Click Start to open the live camera window. Recognition and check-ins "
        "happen there (drag to resize, 'f' = fullscreen, 'q' = quit)."
    )
    if st.button("Start Attendance", type="primary"):
        subprocess.Popen([sys.executable, "attendance.py"])
        st.success("Camera window opened. Check-ins appear on the Dashboard and Reports pages.")


# ====================== REPORTS ======================
elif page == "Reports":
    st.title("Attendance Report")

    selected = st.date_input("Select date", value=date.today())

    session = SessionLocal()
    records = (
        session.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .filter(Attendance.date == selected)
        .order_by(Attendance.check_in)
        .all()
    )
    session.close()

    st.subheader(f"{selected.strftime('%d-%m-%Y')} — {len(records)} present")

    if records:
        rep_df = pd.DataFrame([
            {
                "Employee Code": emp.emp_code,
                "Name": emp.name,
                "Date": selected.strftime("%d-%m-%Y"),
                "Check-in Time": att.check_in.strftime("%H:%M:%S") if att.check_in else "",
                "Status": status_badge(att.check_in),
            }
            for att, emp in records
        ])
        render_table(rep_df)

        st.write("")
        buf = io.BytesIO()
        rep_df.to_excel(buf, index=False, sheet_name="Attendance")
        buf.seek(0)
        st.download_button(
            "Download Excel",
            data=buf,
            file_name=selected.strftime("%d-%m-%Y") + ".xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No check-ins for this date.")

    st.divider()
    st.subheader("Reset attendance")
    r1, r2 = st.columns(2)
    if r1.button(f"Delete {selected.strftime('%d-%m-%Y')} records"):
        session = SessionLocal()
        deleted = session.query(Attendance).filter_by(date=selected).delete()
        session.commit()
        session.close()
        st.success(f"Deleted {deleted} record(s) for {selected.strftime('%d-%m-%Y')}.")

    if r2.button("RESET ALL attendance", type="primary"):
        session = SessionLocal()
        deleted = session.query(Attendance).delete()
        session.commit()
        session.close()
        st.success(f"Reset complete — deleted {deleted} record(s).")


# ====================== HR ASSISTANT ======================
elif page == "HR Assistant":
    api_key = os.environ.get("HR_chatbot_API", "").strip()
    if not api_key:
        st.error("HR_chatbot_API not found in your .env file.")
        st.stop()

    if "hr_msgs" not in st.session_state:
        st.session_state.hr_msgs = []

    def ask_hr(question):
        st.session_state.hr_msgs.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            context = build_hr_context()
            reply = hr_answer(api_key, st.session_state.hr_msgs[:-1], question, context)
        st.session_state.hr_msgs.append({"role": "assistant", "content": reply})

    if st.session_state.hr_msgs:
        h1, h2 = st.columns([4, 1])
        h1.markdown("#### HR Assistant")
        if h2.button("New chat", width="stretch"):
            st.session_state.hr_msgs = []
            st.rerun()

    if not st.session_state.hr_msgs:
        st.markdown(
            "<div class='greet'><h2>How can I help you today?</h2>"
            "<p>Ask about attendance, register or delete an employee, or start attendance.</p></div>",
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        if s1.button("Who's late today?", width="stretch"):
            ask_hr("Who was late today?"); st.rerun()
        if s2.button("Start attendance", width="stretch"):
            ask_hr("Start taking attendance now."); st.rerun()
        if s3.button("Register new employee", width="stretch"):
            ask_hr("I want to register a new employee."); st.rerun()
    else:
        for m in st.session_state.hr_msgs:
            if m["role"] == "user":
                st.markdown(
                    f"<div class='uwrap'><div class='ubub'>{_html.escape(m['content'])}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(m["content"])
                st.markdown("<div style='margin-bottom:18px'></div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask the HR assistant…")
    if prompt:
        ask_hr(prompt)
        st.rerun()