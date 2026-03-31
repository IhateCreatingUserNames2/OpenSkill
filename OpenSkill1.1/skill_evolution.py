"""
skill_evolution.py — Trace2Skill Evolution Engine
===================================================
Implements parallel fleet-based skill evolution from execution trajectories.

Core idea (from Trace2Skill paper, arXiv:2603.25158):
  Skills should not be updated sequentially from single trajectories (which
  causes overfitting to trajectory-local lessons). Instead:
    1. Collect a POOL of N execution trajectories
    2. Dispatch a PARALLEL FLEET of sub-agents to analyze batches
    3. Each sub-agent proposes a PATCH (structured diff to the skill)
    4. HIERARCHICAL CONSOLIDATION merges all patches into one coherent update
    5. Apply the consolidated patch to produce the evolved skill

The fleet approach induces generalizable patterns by identifying lessons
that recur across MULTIPLE trajectories — single-trajectory lessons are
noise, cross-trajectory patterns are signal.

Key difference from MemCollab:
  MemCollab  → contrastive (2 agents, 1 task, extract what's invariant)
  Trace2Skill → inductive  (N trajectories, find prevalent patterns, evolve)
  Together   → MemCollab creates the skill; Trace2Skill evolves it over time

Patch schema (JSON, applied as structured diff to Skill.md):
  {
    "section": "Reasoning Invariants" | "Violation Patterns" | "Normative Constraints" | ...,
    "op":      "insert" | "replace" | "append" | "remove",
    "target":  "existing bullet or section header to anchor",
    "content": "new content to add",
    "justification": "trajectory pattern that motivates this patch",
    "prevalence": 0.0–1.0  # fraction of analyzed trajectories that triggered this
  }
"""

import json
import re
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Trace2Skill fleet configuration
FLEET_BATCH_SIZE   = 3   # Trajectories per sub-agent (paper uses ~5-10)
MAX_FLEET_SIZE     = 8   # Max parallel sub-agents
MIN_PREVALENCE     = 0.3 # Only keep patches seen in >30% of trajectories
MERGE_LEVELS       = 2   # Hierarchical merge levels (paper uses 4 for 323 patches)


# ── LLM Helper ───────────────────────────────────────────────────────────────

async def _call(api_key: str, model: str, messages: list, max_tokens: int = 2000) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openskill.local",
        "X-Title": "OpenSkill",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        return content.strip()


def _strip_think(text: str) -> str:
    return re.sub(r'<(?:think|thought)>.*?(?:</(?:think|thought)>|$)', '', text,
                  flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_json(text: str) -> Optional[list | dict]:
    """Robustly extract first JSON object or array from text."""
    # Try markdown-fenced first
    m = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try naked JSON
    for pattern in [r'(\[.*\])', r'(\{.*\})']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


# ── Stage 2: Sub-Agent Patch Proposal ────────────────────────────────────────

async def _propose_patches_for_batch(
    api_key: str,
    model: str,
    skill_md: str,
    trajectories: list[dict],  # [{"task": str, "trajectory": str, "success": bool}]
    batch_idx: int,
) -> list[dict]:
    """
    Sub-agent: analyze a batch of trajectories and propose patches.

    Trace2Skill Stage 2: Error analyst + Success analyst sub-agents.
    Each sub-agent independently proposes patches without knowing what
    other sub-agents are analyzing.
    """
    # Separate successes from failures for targeted analysis
    successes = [t for t in trajectories if t.get("success", True)]
    failures  = [t for t in trajectories if not t.get("success", True)]

    traj_text = ""
    for i, t in enumerate(trajectories):
        status = "SUCCESS" if t.get("success", True) else "FAILURE"
        traj_text += f"\n---TRAJECTORY {i+1} [{status}]---\n"
        traj_text += f"Task: {t.get('task', '')[:200]}\n"
        traj_text += f"Execution: {t.get('trajectory', '')[:600]}\n"

    system = (
        "You are a Skill Evolution Sub-Agent (Trace2Skill framework).\n"
        "Analyze execution trajectories and propose targeted patches to improve a skill document.\n\n"
        "PATCH RULES:\n"
        "1. Only propose patches backed by OBSERVABLE patterns in the trajectories\n"
        "2. Patches must improve the skill for FUTURE executions, not fix past errors\n"
        "3. Each patch targets a specific section of the skill document\n"
        "4. Estimate prevalence: fraction of trajectories that motivated this patch\n"
        "5. Be concrete: 'avoid X; enforce Y' format for constraints\n"
        "6. DO NOT introduce speculative or unverifiable changes\n\n"
        "Allowed sections: 'Reasoning Invariants', 'Violation Patterns', "
        "'Normative Constraints', 'When to Apply', 'Example Pattern'\n\n"
        "Allowed ops: 'append' (add new bullet), 'replace' (update existing), "
        "'insert' (add before anchor), 'remove' (delete obsolete bullet)\n\n"
        "Output ONLY a JSON array of patch objects. No preamble.\n"
        "Schema: [{\"section\": str, \"op\": str, \"target\": str, "
        "\"content\": str, \"justification\": str, \"prevalence\": float}]"
    )

    user = (
        f"CURRENT SKILL DOCUMENT:\n```\n{skill_md[:2000]}\n```\n\n"
        f"EXECUTION TRAJECTORIES (batch {batch_idx + 1}):\n{traj_text}\n\n"
        f"Statistics: {len(successes)} successes, {len(failures)} failures.\n"
        "Propose patches that generalize across these trajectories. "
        "Focus on patterns that appear in MULTIPLE trajectories."
    )

    raw = await _call(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], max_tokens=1500)

    cleaned = _strip_think(raw)
    patches = _extract_json(cleaned)
    if not isinstance(patches, list):
        return []

    # Validate and filter patches
    valid = []
    for p in patches:
        if not isinstance(p, dict):
            continue
        if p.get("section") and p.get("op") and p.get("content"):
            p.setdefault("prevalence", 0.5)
            p.setdefault("target", "")
            p.setdefault("justification", "")
            valid.append(p)
    return valid


# ── Stage 3: Hierarchical Patch Consolidation ─────────────────────────────────

async def _merge_patch_groups(
    api_key: str,
    model: str,
    patch_groups: list[list[dict]],
    skill_md: str,
    level: int,
) -> list[dict]:
    """
    Merge operator (Trace2Skill Stage 3): consolidate multiple patch groups
    into a single coherent patch set.

    Hierarchical: call this recursively until one group remains.
    At each level, the merge operator:
      1. Deduplicates redundant patches
      2. Resolves conflicts (keep stronger justification)
      3. Identifies prevalent patterns (recur across groups → systemic property)
      4. Raises prevalence for cross-group patterns
    """
    all_patches = [p for group in patch_groups for p in group]

    if len(all_patches) <= 3:
        return all_patches  # Already consolidated

    patches_json = json.dumps(all_patches[:40], indent=2)[:4000]  # Token budget

    system = (
        "You are a Skill Edit Coordinator (Trace2Skill merge operator).\n"
        "Receive independently-proposed patches and merge into one coherent, non-redundant set.\n\n"
        "MERGE RULES:\n"
        "1. Deduplicate: When multiple patches propose similar edits, keep the most specific/best-worded\n"
        "2. Resolve conflicts: If patches contradict each other, choose the one with stronger justification\n"
        "3. Preserve unique insights: Include all unique, non-redundant patches\n"
        "4. PREVALENT PATTERN BIAS: When multiple patches address the SAME failure class, "
        "   treat this as a systemic property — raise its prevalence and express as a general principle\n"
        "5. Conciseness: merged result must have ≤ sum of unique patches across all inputs\n\n"
        "Output ONLY a JSON array of merged patches. Same schema as input. No preamble."
    )

    user = (
        f"SKILL DOCUMENT (context):\n```\n{skill_md[:800]}\n```\n\n"
        f"PATCHES TO MERGE (level {level}):\n{patches_json}\n\n"
        "Merge into a coherent, deduplicated patch set. "
        "Elevate prevalent patterns to general principles."
    )

    raw = await _call(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], max_tokens=2000)

    cleaned = _strip_think(raw)
    merged = _extract_json(cleaned)
    if not isinstance(merged, list):
        return all_patches  # Fallback: return unmerged
    return merged


async def _hierarchical_consolidate(
    api_key: str,
    model: str,
    all_patch_groups: list[list[dict]],
    skill_md: str,
) -> list[dict]:
    """
    Hierarchical consolidation of all sub-agent patch groups.

    Trace2Skill uses 4 merge levels for 323 patches.
    We use 2 levels for typical library sizes (< 50 skills).

    Level 1: Merge in pairs
    Level 2: Final global merge
    """
    if len(all_patch_groups) == 0:
        return []
    if len(all_patch_groups) == 1:
        return all_patch_groups[0]

    current_groups = all_patch_groups

    for level in range(MERGE_LEVELS):
        if len(current_groups) <= 1:
            break
        # Pair up groups
        merged_groups = []
        for i in range(0, len(current_groups), 2):
            if i + 1 < len(current_groups):
                merged = await _merge_patch_groups(
                    api_key, model,
                    [current_groups[i], current_groups[i + 1]],
                    skill_md, level=level + 1
                )
                merged_groups.append(merged)
            else:
                merged_groups.append(current_groups[i])  # Odd group carries over
        current_groups = merged_groups

    return current_groups[0] if current_groups else []


# ── Patch Application ─────────────────────────────────────────────────────────

def apply_patches_to_skill(skill_md: str, patches: list[dict]) -> str:
    """
    Apply consolidated patches to a skill Markdown document.

    Supports: append, replace, insert, remove operations.
    Operations are applied in order; each targets a section by header name.
    """
    # Filter by prevalence threshold
    strong_patches = [p for p in patches if p.get("prevalence", 0) >= MIN_PREVALENCE]

    if not strong_patches:
        return skill_md

    lines = skill_md.split("\n")
    result = lines[:]

    for patch in strong_patches:
        section = patch.get("section", "")
        op      = patch.get("op", "")
        target  = patch.get("target", "")
        content = patch.get("content", "")

        # Find section header in document
        section_line = -1
        for i, line in enumerate(result):
            if f"## {section}" in line or f"### {section}" in line:
                section_line = i
                break

        if section_line == -1:
            # Section not found — append at end of document for "append" ops
            if op == "append" and content:
                result.append(f"\n## {section}\n\n- {content}")
            continue

        # Find end of section (next ## header or end of file)
        section_end = len(result)
        for i in range(section_line + 1, len(result)):
            if result[i].startswith("## ") or result[i].startswith("---"):
                section_end = i
                break

        section_content = result[section_line:section_end]

        if op == "append":
            # Add new bullet at end of section
            insert_pos = section_end
            # Find last non-empty line in section
            for i in range(section_end - 1, section_line, -1):
                if result[i].strip():
                    insert_pos = i + 1
                    break
            result.insert(insert_pos, f"- {content}")

        elif op == "replace" and target:
            # Replace existing bullet matching target text
            for i in range(section_line, section_end):
                if target.lower() in result[i].lower():
                    result[i] = f"- {content}"
                    break

        elif op == "insert" and target:
            # Insert before line matching target
            for i in range(section_line, section_end):
                if target.lower() in result[i].lower():
                    result.insert(i, f"- {content}")
                    break

        elif op == "remove" and target:
            # Remove bullet matching target
            result = [line for i, line in enumerate(result)
                      if not (section_line <= i < section_end and target.lower() in line.lower())]

    # Add evolution changelog entry
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    changelog = (
        f"\n---\n\n## Evolution Log\n\n"
        f"*Evolved {now} via Trace2Skill — "
        f"{len(strong_patches)} patches applied (prevalence ≥ {MIN_PREVALENCE})*\n"
    )
    if "## Evolution Log" in skill_md:
        # Update existing log
        log_start = next((i for i, l in enumerate(result) if "## Evolution Log" in l), -1)
        if log_start != -1:
            result = result[:log_start]
    result.append(changelog)

    return "\n".join(result)


# ── Stage 1: Trajectory Generation ───────────────────────────────────────────

async def generate_evolution_trajectories(
    api_key: str,
    model: str,
    skill_md: str,
    tasks: list[str],
) -> list[dict]:
    """
    Run the skill-guided agent on a list of tasks to generate trajectories.
    Stage 1 of Trace2Skill.

    Returns list of trajectory dicts with success labels.
    Uses the CURRENT skill as context (so evolution is skill-conditioned).
    """
    async def run_one(task: str) -> dict:
        system = (
            "You are a reasoning agent. Use the provided SKILL GUIDE to solve the task.\n"
            "Show your full reasoning. At the end, state: RESULT: [SUCCESS|FAILURE] and why.\n\n"
            f"SKILL GUIDE:\n{skill_md[:1500]}"
        )
        try:
            raw = await _call(api_key, model, [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Task:\n{task}"},
            ], max_tokens=2000)
            success = "RESULT: SUCCESS" in raw.upper() or "RESULT:SUCCESS" in raw.upper()
            return {"task": task, "trajectory": raw, "success": success}
        except Exception as e:
            return {"task": task, "trajectory": f"Error: {e}", "success": False}

    # Run trajectories in parallel (bounded concurrency)
    sem = asyncio.Semaphore(4)
    async def bounded(task):
        async with sem:
            return await run_one(task)

    results = await asyncio.gather(*[bounded(t) for t in tasks])
    return list(results)


# ── Main Evolution Pipeline ───────────────────────────────────────────────────

async def evolve_skill(
    api_key: str,
    analyst_model: str,
    skill_md: str,
    trajectories: list[dict],
) -> dict:
    """
    Full Trace2Skill evolution pipeline.

    Stage 1: Trajectories received (pre-generated or passed in)
    Stage 2: Fleet of sub-agents analyze batches in parallel → patches
    Stage 3: Hierarchical consolidation → final patch set
    Stage 4: Apply patches → evolved skill

    Returns:
    {
      "evolved_md":     str,   — updated Skill.md
      "patches_applied": list, — consolidated patches
      "patch_count":    int,
      "fleet_size":     int,
      "success_rate":   float,
    }
    """
    if not trajectories:
        return {"evolved_md": skill_md, "patches_applied": [], "patch_count": 0,
                "fleet_size": 0, "success_rate": 0.0}

    success_rate = sum(1 for t in trajectories if t.get("success")) / len(trajectories)

    # Stage 2: Dispatch parallel fleet
    # Divide trajectories into batches
    batch_size = max(1, min(FLEET_BATCH_SIZE, len(trajectories)))
    batches = [
        trajectories[i:i + batch_size]
        for i in range(0, len(trajectories), batch_size)
    ]
    # Cap fleet size
    batches = batches[:MAX_FLEET_SIZE]

    # Parallel sub-agent patch proposals
    tasks = [
        _propose_patches_for_batch(api_key, analyst_model, skill_md, batch, idx)
        for idx, batch in enumerate(batches)
    ]
    patch_groups = await asyncio.gather(*tasks)
    patch_groups = [g for g in patch_groups if g]  # Remove empty groups

    # Stage 3: Hierarchical consolidation
    consolidated = await _hierarchical_consolidate(
        api_key, analyst_model, list(patch_groups), skill_md
    )

    # Stage 4: Apply to skill document
    evolved_md = apply_patches_to_skill(skill_md, consolidated)

    return {
        "evolved_md":      evolved_md,
        "patches_applied": consolidated,
        "patch_count":     len([p for p in consolidated if p.get("prevalence", 0) >= MIN_PREVALENCE]),
        "fleet_size":      len(patch_groups),
        "success_rate":    success_rate,
    }
