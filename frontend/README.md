# RAGMed frontend

The web app patients use. Not started yet; this folder reserves its place so the frontend and the
AI backend stay fully separate.

## What it will be

- **Next.js** (App Router, TypeScript) with Tailwind and shadcn/ui, deployed to Vercel.
- A chat that streams the agent's answer, shows the cited sources, and lets the patient pick and
  confirm an appointment slot.
- A thin server-side route (`/api/chat`) that forwards messages to the backend's Care Coordinator
  agent over A2A. The browser never talks to the backend directly, so backend keys stay secret.

## How it connects to the backend

```text
browser ──▶ Next.js route /api/chat (Vercel) ──A2A──▶ backend Care Coordinator agent
```

The frontend depends only on the A2A protocol, not on backend code, so either side can be changed
or redeployed on its own.
