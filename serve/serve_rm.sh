#!/bin/bash

# Start the vLLM server with the specified model and parameters
vllm serve amphora/dprm-ckpt1 --dtype auto --api-key token-abc123 --port 1210
