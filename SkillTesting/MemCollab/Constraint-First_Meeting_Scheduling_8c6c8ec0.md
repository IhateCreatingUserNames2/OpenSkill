---
name: Constraint-First Meeting Scheduling
domain: Reasoning / Planning / Resource Allocation
generated: 2026-04-02 08:30 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Constraint-First Meeting Scheduling

> A systematic method for scheduling meetings into rooms and time slots by resolving the most restrictive constraints first, verifying each constraint explicitly before declaring a solution valid.

## When to Apply

Apply this skill whenever scheduling N items into M rooms across T time slots with a mix of simultaneity, sequencing, exclusion, and resource constraints — especially when constraints interact and must be resolved in dependency order.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Meetings sharing the same time slot must occupy different rooms; simultaneity never implies co-location.
- Resource-exclusive constraints (e.g., projector only in Room 1) must be resolved before placing any other meetings that could block that resource.
- Sequential constraints (A before C) must be resolved immediately when the number of available slots is small, pinning each meeting to a specific slot without ambiguity.
- Same-time constraints (A and B simultaneous) propagate downstream: once one meeting's slot is fixed, the other's slot is fixed by inheritance.
- Every constraint must be explicitly cross-checked against the final schedule table before the solution is declared valid.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Assuming 'same time' means 'same room' for two meetings that must occur simultaneously.
- ⚠️ Leaving sequential placement (A before C) ambiguous and resolving it only after other placements have already constrained the options.
- ⚠️ Declaring all constraints satisfied without explicitly verifying each one against the final room-and-slot assignments.
- ⚠️ Assigning a resource-dependent meeting's required room to another meeting before confirming the resource-dependent meeting can still be accommodated.
- ⚠️ Filling the last remaining slot by elimination without explicitly checking the 'not same room' constraint for the relevant meeting pair.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce resolving resource-exclusive room assignments first (e.g., projector → Room 1 for E), then fit all other meetings around that fixed placement; avoid assigning that room to others before the resource-dependent meeting is placed.
- Enforce pinning sequentially ordered meetings to specific slots immediately when slots are limited (A → slot 1, C → slot 2); avoid deferring this decision.
- Enforce placing simultaneously required meetings in the same time slot but different rooms; avoid conflating time-slot sharing with room sharing.
- Enforce propagating slot assignments through dependency chains (A fixed to slot 1 → B also fixed to slot 1 via same-time constraint); avoid treating each constraint in isolation.
- Enforce a final explicit constraint audit by iterating through every stated constraint and checking it against the completed schedule table; avoid premature success declarations.

---

## Example Pattern

```
Given meetings {A, B, C, D, E}, rooms {R1, R2, R3}, slots {S1, S2}, and constraints [E→R1, A before C, A‖B simultaneous, B≠D same room]: (1) Place E in R1 at some slot. (2) Pin A to S1 and C to S2. (3) Place B in S1 in a different room from A. (4) Place D in a slot and room ≠ B's room. (5) Verify all five constraints against the final table explicitly.
```

---

## Source: Contrastive Trajectory Analysis

### Task
```
Schedule 5 meetings (A, B, C, D, E) into 3 rooms (Room 1, Room 2, Room 3) between 9 AM and 11 AM. Each meeting is 1 hour long.
Constraints:
1. A must happen before C.
2. B and D cannot be in the same room.
3. E requires a projector (only Room 1 has one).
4. A and B must happen at the same time.
Provide a valid schedule and explain how constraints are satisfied.
```

### Weak Agent Trajectory (`meta-llama/llama-3.1-8b-instruct`)
<details>
<summary>Expand weak agent reasoning</summary>

To tackle this problem, I will use a systematic approach to schedule the meetings while satisfying all the constraints.

## Step 1: Analyze the constraints
First, let's list all the given constraints and understand their implications:
1. A must happen before C.
2. B and D cannot be in the same room.
3. E requires a projector, which is only available in Room 1.
4. A and B must happen at the same time.

## Step 2: Determine the scheduling window
The meetings are scheduled between 9 AM and 11 AM. Since each meeting is 1 hour long, we have 2 hours to schedule 5 meetings.

## Step 3: Schedule meeting A and B
Since A and B must happen at the same time, we can schedule them first. To satisfy constraint 1 (A must happen before C), A should be scheduled before C.

## Step 4: Choose the room for meeting A and B
Since A and B occur at the same time, they can either be in the same room or different rooms. However, to satisfy constraint 4, the only option is to have them in the same room. We can choose Room 1 for A and B because Room 1 is the only room that has a projector, which might be needed for meeting E. This leaves Room 2 and Room 3 for the remaining meetings.

## Step 5: Schedule meeting E
Meeting E requires a projector, which is only available in Room 1. Since A and B are already scheduled in Room 1, we can schedule E in Room 1 after A and B, but E cannot be scheduled before A and B since there is no way to fit it in the time slot before A and B without violating the constraints.

## Step 6: Schedule meeting C
Meeting C must happen after A. Given that A and B are scheduled in Room 1, and E requires this room, we cannot schedule C in Room 1. Therefore, C must be scheduled in Room 2 or Room 3. We choose Room 2 for C because it's the first available room that meets the constraint of being after A.

## Step 7: Schedule meeting D
Meeting D cannot be in the same room as B, and since B is in Room 1, D cannot be in Room 1. Given that we've already scheduled A and C, we have Room 2 and Room 3 left. Since C is in Room 2, D can only be in Room 3.

## Step 8: Finalize the schedule
We have the following schedule:
- 9:00-10:00 AM: A and B in Room 1
- 10:00-11:00 AM: E in Room 1 (since A and B finish, E can fit)
- 9:00-10:00 AM: C in Room 2
- 10:00-11:00 AM: D in Room 3

## Step 9: Check if all constraints are satisfied
1. A happens before C.
2. B and D are not in the same room.
3. E requires a projector in Room 1.
4. A and B happen at the same time.

The final answer is:

**Schedule:**

- 9:00-10:00 AM: A and B in Room 1
- 10:00-11:00 AM: E in Room 1
- 9:00-10:00 AM: C in Room 2
- 10:00-11:00 AM: D in Room 3

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

# Meeting Scheduling Solution

## Analysis of Constraints

Let me work through this systematically.

**Available slots:** 9–10 AM and 10–11 AM
**Rooms:** Room 1, Room 2, Room 3
**Meetings:** A, B, C, D, E (each 1 hour)

---

## Step-by-Step Reasoning

**Step 1: Handle Constraint 3 (E needs projector → Room 1)**
> E must be in Room 1 at either 9 AM or 10 AM.

**Step 2: Handle Constraint 4 (A and B at the same time)**
> A and B must share the same time slot (but different rooms).

**Step 3: Handle Constraint 1 (A before C)**
> A must be in the **9–10 AM** slot, and C must be in the **10–11 AM** slot.

**Step 4: Since A is at 9 AM → B is also at 9 AM** (from Constraint 4)

**Step 5: Place E in Room 1**
> Since A and B occupy two rooms at 9 AM, E can go in Room 1 at **10–11 AM** (leaving 9 AM slots for A and B in Rooms 2 and 3).

**Step 6: Handle Constraint 2 (B and D not in same room)**
> B is in one room; D must be placed in a *different* room. D goes in the remaining slot.

---

## ✅ Final Schedule

| Time Slot | Room 1 (Projector) | Room 2 | Room 3 |
|------------|--------------------|--------|--------|
| **9–10 AM** | *(empty)* | **A** | **B** |
| **10–11 AM** | **E** | **C** | **D** |

---

## Constraint Verification

| # | Constraint | Status |
|---|------------|--------|
| 1 | **A before C** | ✅ A is at 9 AM, C is at 10 AM |
| 2 | **B and D not in same room** | ✅ B is in Room 3, D is in Room 3... |

> ⚠️ *Correction:* Let me swap D to Room 2 and C to Room 3:

| Time Slot | Room 1 (Projector) | Room 2 | Room 3 |
|------------|--------------------|--------|--------|
| **9–10 AM** | *(empty)* | **A** | **B** |
| **10–11 AM** | **E** | **D** | **C** |

| # | Constraint | Status |
|---|------------|--------|
| 1 | **A before C** | ✅ A at 9 AM (Room 2), C at 10 AM (Room 3) |
| 2 | **B and D not in same room** | ✅ B in Room 3, D in Room 2 — different rooms |
| 3 | **E needs projector (Room 1)** | ✅ E is in Room 1 |
| 4 | **A and B same time** | ✅ Both at 9–10 AM |

---

## Summary
All **5 constraints are satisfied** with this clean two-slot arrangement. The key insight was that A must be in the **first slot** (to precede C), which naturally anchored B at 9 AM as well, leaving the 10 AM slot for C, D, and E.

</details>

### Raw Extracted Constraints
```
1. When scheduling meetings that must occur simultaneously (same time constraint), enforce placing them in different rooms since one room cannot host two meetings at once; avoid assuming "same time" means "same room."

2. When a sequential constraint (A before C) exists with only two time slots available, enforce assigning the first meeting to the earlier slot and the dependent meeting to the later slot immediately; avoid leaving this placement ambiguous until later steps.

3. When verifying a proposed schedule against all constraints, enforce explicitly checking each constraint against the final table before declaring success; avoid prematurely claiming all constraints are satisfied without cross-referencing room assignments.

4. When a resource constraint limits a meeting to one specific room (e.g., projector in Room 1), enforce resolving that placement first and then fitting other meetings around it; avoid assigning that room to other meetings without confirming the resource-dependent meeting can still be accommodated.

5. When placing the final remaining meetings into slots and rooms, enforce checking the "not same room" constraint by comparing the room of the constrained meeting pair explicitly; avoid assigning by elimination without verifying the constraint is actually satisfied in the resulting configuration.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
