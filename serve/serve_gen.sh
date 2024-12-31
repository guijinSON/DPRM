#!/bin/bash

# Start the vLLM server with the specified model and parameters
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto --api-key token-abc123 --port 1210
