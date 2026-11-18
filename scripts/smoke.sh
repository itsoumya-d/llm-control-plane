#!/usr/bin/env bash
set -euo pipefail

echo "Running LLM Control Plane smoke check..."
python -c "import llm_control_plane; print('llm_control_plane module load OK')"
python -c "from llm_control_plane.workforce_agent_browser_orchestrator import WorkforceBrowserOrchestrator; print('workforce orchestrator import OK')"
echo "Smoke check passed successfully."
