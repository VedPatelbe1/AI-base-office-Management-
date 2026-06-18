# FaceTrack — Face Recognition Attendance

Office attendance system using an IP camera, InsightFace recognition, a Streamlit
dashboard, and an OpenAI-powered HR assistant that can take actions.

## Setup
1. Create a virtual environment and install deps:


2. Copy `.env.example` to `.env` and add your OpenAI key.
3. Set your camera RTSP URL in `config.py`.
4. Initialise the database:

5. Run the dashboard:

## Notes
- `attendance.db` and `.env` are gitignored (biometric data + secrets).