FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# UI only needs HTTP client + visualization libs; intentionally lean.
COPY docker/ui-requirements.txt ./ui-requirements.txt
RUN pip install --upgrade pip && pip install -r ui-requirements.txt

COPY eeie/ui_streamlit ./eeie/ui_streamlit
COPY .streamlit ./.streamlit
# Empty package marker so Streamlit can `import eeie.ui_streamlit.lib`
RUN touch eeie/__init__.py

EXPOSE 8501

CMD ["streamlit", "run", "eeie/ui_streamlit/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
