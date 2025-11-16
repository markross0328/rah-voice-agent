# RAH Voice Agent (Production-Oriented Skeleton)

This repo is structured for a production-style AI voice/phone agent for a Right at Home–style franchise.

Key pieces:
- apps/agent: LiveKit + LLM agent, tools, guards, RAG, routing
- services/*: domain microservices (WellSky adapter, scheduler, notifier)
- packages/*: shared libraries (logging, audit, http clients)
- data/*: company/scenario/example knowledge bases (no PHI)
- offline/*: post-call analysis, datasets, repro harness
- storage/*: SQL schemas for analytics + audit logs

Quick start (after you fill code in):
1. cp config/env.example config/.env
2. docker compose -f config/compose.yaml up --build
3. Connect a LiveKit web client to LIVEKIT_URL and test the agent.