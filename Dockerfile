FROM python:3.7-slim

RUN mkdir -p /tornado

COPY . /tornado

WORKDIR /tornado

RUN pip install -r requirement.txt

EXPOSE 8000

CMD ["python", "-u", "github_prequential_test.py"]