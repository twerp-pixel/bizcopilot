# Setup, Run, and Deploy Guide

Written for EndeavourOS / Arch (8GB RAM) but notes for other OSes are included.

## 1. Local setup

```bash
# system packages (Arch/EndeavourOS) - python3-venv equivalent is built into python
sudo pacman -S python python-pip git --needed

cd bizcopilot
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Notes for 8GB RAM:
- `sentence-transformers` pulls in PyTorch (CPU build). The download is a few
  hundred MB and inference on a 22-doc corpus is negligible RAM — this is
  fine on 8GB. Don't run a local LLM (7B+ params) alongside this; we use the
  Anthropic API for generation specifically so your machine only ever holds
  the small embedding model in memory.
- If `pip install` is slow/flaky on Arch, make sure you're in the venv
  (`which python3` should point into `.venv/`) and not fighting pacman's
  system Python packages.

## 2. API key

```bash
cp .env.example .env
```
Edit `.env` and add your `ANTHROPIC_API_KEY` (or adapt `agent/llm_client.py`
to call OpenAI instead — it's one function).

## 3. Generate the synthetic data

```bash
python3 data/generate_data.py
```
This creates `data/bizcopilot.db` and the doc corpus in `data/docs/`. Idempotent —
re-run any time to reset the data.

## 4. Run it

```bash
./run.sh
```
Then open **http://localhost:8000** — you'll see the chat UI. Try the two
example questions on the page first; they're guaranteed to hit the two
richest injected scenarios in the data.

Or run pieces individually while developing:
```bash
python3 tools/sql_tool.py                 # sanity check SQL tool
python3 tools/retrieval_tool.py           # sanity check retrieval
python3 -m agent.orchestrator "Why did SKU-1004 sales drop in the Midwest around week 9?"
```

## 5. Run the evaluation

```bash
# retrieval-only, no API key needed, fast and free — run this while iterating
python3 -m eval.eval_harness --retrieval-only

# full end-to-end (costs a handful of API calls)
python3 -m eval.eval_harness
```
Copy the printed Recall@K / MRR / groundedness numbers into the README table.
**Use your real numbers, not placeholders**, before you put this on your resume.

## 6. Push to GitHub

```bash
cd bizcopilot
git init
git add .
git commit -m "Initial commit: BizCopilot agentic business analyst copilot"

# create a new repo on github.com first (via the web UI, "New repository"),
# then:
git branch -M main
git remote add origin https://github.com/<your-username>/bizcopilot.git
git push -u origin main
```
`.env` is already in `.gitignore` — double check it never gets committed
(`git status` should not show it).

Write a good GitHub README top section — the one already in the repo has
the business framing up front, which matters more than you'd think for a
portfolio repo; recruiters skim.

## 7. Make it live (deploy it)

You have two realistic free options. I'd go with **Hugging Face Spaces**
first — it's the fastest, and "an AI project I deployed to HF Spaces" reads
well on a portfolio for an AI-focused role.

### Option A: Hugging Face Spaces (recommended, free, simplest)
1. Create a free account at huggingface.co if you don't have one.
2. Create a new Space → choose **Docker** as the SDK (not Gradio/Streamlit —
   we already have a Dockerfile, so this just works).
3. Either:
   - Push your GitHub repo's contents into the Space repo (Spaces are git
     repos too — `git remote add space https://huggingface.co/spaces/<you>/bizcopilot`
     then `git push space main`), or
   - Use the Spaces web UI to upload the files directly.
4. In the Space's **Settings → Repository secrets**, add `ANTHROPIC_API_KEY`
   (never put it in a committed file).
5. The Space will build the Dockerfile and expose it on port 7860
   automatically (already set in the Dockerfile). Build takes a few minutes
   the first time (downloading the embedding model dependencies).
6. You get a public URL like `https://huggingface.co/spaces/<you>/bizcopilot`
   — put this directly on your resume/LinkedIn.

### Option B: Render.com (also free tier, works well for FastAPI)
1. Push the repo to GitHub first (step 6 above).
2. On render.com: New → Web Service → connect your GitHub repo.
3. Render auto-detects the Dockerfile. Set the environment variable
   `ANTHROPIC_API_KEY` in the dashboard's Environment tab.
4. Deploy. Render gives you a `https://<name>.onrender.com` URL.
5. Note: Render's free tier spins the service down after inactivity, so the
   first request after idle takes ~30-60s to wake up — mention this if you
   demo it live so it's not confusing.

### What I can't do for you
I can't create accounts, add billing/API keys, or push to a remote host on
your behalf from here — those steps need your own credentials. Everything
above is what remains once you have the code, which I've already written,
tested for syntax, and verified runs correctly for the parts that don't
need network access (data generation, SQL tool, stats tool all ran
successfully in the build environment).

## 8. Before you put it on your resume
- Actually run the eval harness and put your real numbers in the README —
  don't estimate them.
- Test the two example questions in the UI and read the memo output —
  make sure the grounding verifier is actually catching bad claims (you can
  temporarily break something in the synthesizer prompt to confirm the
  verifier flags it, then put it back).
- Take a screenshot or short screen recording of the working UI — useful
  for a portfolio site or LinkedIn post even if you don't keep the Space
  running forever.
