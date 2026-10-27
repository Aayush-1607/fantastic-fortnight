@echo off
cd /d %~dp0
pip install -r requirements.txt
cd backend
python app.py
