#!/bin/bash
uvicorn manage:app --port 9999 --reload --host 0.0.0.0
