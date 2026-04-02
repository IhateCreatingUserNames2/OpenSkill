---
name: Constraint-Propagating Meeting Scheduler
domain: Reasoning / Combinatorial Planning / Resource Allocation
generated: 2026-04-02 09:08 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Constraint-Propagating Meeting Scheduler

> A systematic skill for scheduling multiple meetings into limited rooms and time slots by resolving hard constraints in dependency order, propagating each assignment's implications immediately, and verifying every constraint explicitly before declaring a schedule valid.

## When to Apply

Apply this skill whenever a scheduling problem involves multiple agents (meetings, tasks, people) competing for limited resources (rooms, time slots, equipment) under a combination of ordering, simultaneity, exclusivity, and resource-assignment constraints.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Resource-exclusive constraints (a meeting requiring a specific room) must be resolved first, as they eliminate the most options and anchor the rest of the schedule.
- Ordering constraints (A before C) must be translated into concrete time-slot assignments before room assignments are made.
- Simultaneity constraints (A and B at the same time) must derive their shared slot from any ordering constraints already applied to either meeting, not chosen independently.
- Every constraint must be checked explicitly against the final schedule in a dedicated verification pass, even when the schedule was built by constraint-driven construction.
- When meetings compete for the same slot or room, unconstrained meetings are reassigned first before disturbing already-anchored constrained meetings.
- Each assignment must immediately propagate its implications to all remaining unscheduled meetings before the next assignment is made.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Placing a resource-constrained meeting (e.g., E requiring Room 1) without first anchoring it, then discovering a conflict only after filling other slots.
- ⚠️ Treating the time placements of simultaneously-constrained meetings (A and B) as independent decisions.
- ⚠️ Reversing or co-slotting ordered meetings (A and C in the same slot, or C before A).
- ⚠️ Placing two room-exclusive meetings (B and D must differ) in the same room without checking the constraint.
- ⚠️ Declaring a schedule valid based solely on the construction process without a final constraint-by-constraint verification pass.
- ⚠️ Resolving a resource conflict by displacing an already-anchored constrained meeting instead of moving an unconstrained one.
- ⚠️ Assigning rooms or slots to later meetings without updating the constraint state after each prior assignment.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce resolving resource-exclusive room constraints first (e.g., E → Room 1) before assigning any other meetings; avoid filling that room with unconstrained meetings before the resource-dependent meeting is placed.
- Enforce translating ordering constraints into time-slot inequalities (A at slot T1, C at slot T2 where T2 > T1) before assigning rooms; avoid assigning rooms to ordered meetings before their relative time positions are fixed.
- Enforce deriving the shared time slot for simultaneously-constrained meetings from any ordering constraints already applied to either of them; avoid treating their time placement as a free independent choice.
- Enforce immediately propagating each assignment's implications (occupied slot, occupied room, derived slot for linked meetings) to all remaining unscheduled meetings after every placement decision; avoid treating placements as isolated steps.
- Enforce resolving slot or room conflicts by reassigning the least-constrained unplaced meeting to an alternative valid position; avoid displacing an already-anchored constrained meeting to resolve a conflict.
- Enforce performing a complete constraint-by-constraint verification pass over the final schedule before declaring it valid; avoid assuming correctness from the construction process alone.

---

## Example Pattern

```
Given meetings {A, B, C, D, E}, rooms {R1, R2, R3}, slots {S1, S2}, with E→R1, A<C, A∥B, B≠D(room): (1) Anchor E in R1 at any slot. (2) Set A=S1, C=S2 from ordering. (3) Set B=S1 from simultaneity with A. (4) Assign A and B to distinct rooms ≠ each other within S1. (5) Place D in a room ≠ B's room. (6) Fill E into remaining R1 slot. (7) Verify all six constraints explicitly against the final table.
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

To solve this problem, we can use a backtracking algorithm. Backtracking is a problem-solving strategy that involves recursively exploring the possible solutions by considering all possible choices and backtracking when a dead-end is reached.

**Step 1: Define the schedule**

Let's represent the schedule as a dictionary where the keys are the rooms and the values are lists of meeting times.

```
schedule = {
    "Room 1": [],
    "Room 2": [],
    "Room 3": []
}
```

**Step 2: Initialize an empty schedule**

Since each meeting is 1 hour long and we want to schedule from 9 AM to 11 AM, we can define a list of available times for each room.

```
available_times = [(9, 10), (10, 11)]
```

**Step 3: Assign meeting A**

Since A must happen before C, we can assign A to either Room 1 or Room 2. However, since B and D cannot be in the same room, we cannot assign A to Room 2 if D is assigned to Room 2. We'll assign A to Room 1 for now.

```
schedule["Room 1"].append((9, 10))  # Assign A to Room 1 from 9-10 AM
```

**Step 4: Assign meeting B**

Since A and B must happen at the same time, we can assign B to Room 2 at the same time as A.

```
schedule["Room 2"].append((9, 10))  # Assign B to Room 2 at 9-10 AM
```

**Step 5: Assign meeting C**

Since A must happen before C, we can assign C to Room 1 from 10-11 AM.

```
schedule["Room 1"].append((10, 11))  # Assign C to Room 1 from 10-11 AM
```

**Step 6: Assign meeting D**

Since B and D cannot be in the same room, we can assign D to Room 3.

```
schedule["Room 3"].append((9, 10))  # Assign D to Room 3 from 9-10 AM
```

**Step 7: Assign meeting E**

Since E requires a projector, which is only available in Room 1, we can assign E to Room 1 from 11-12 AM.

However, this conflicts with C, which is already assigned to Room 1 from 10-11 AM. We need to reschedule E.

```
# Since E requires a projector, assign it to Room 1 from 10-11 AM
schedule["Room 1"].append((10, 11))  # Assign E to Room 1 from 10-11 AM

# Remove C from Room 1 since it conflicts with E
schedule["Room 1"].remove((10, 11))
```

**Step 8: Confirm the schedule**

The final schedule is:

```
{
    "Room 1": [(9, 10), (10, 11)],
    "Room 2": [(9, 10)],
    "Room 3": [(9, 10)]
}
```

The constraints are satisfied:

* A happens before C.
* B and D are not in the same room.
* E is in a room with a projector.
* A and B happen at the same time.

Here's the final answer:

The valid schedule is:

| Time | Room 1 | Room 2 | Room 3 |
| --- | --- | --- | --- |
| 9:00-10:00 | A | B | D |
| 10:00-11:00 | E |  |  |
| 11:00-12:00 |  |  |  |

This schedule satisfies all the constraints.

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

**Step 1: Handle Constraint 3 (E needs Room 1)**
→ E must be in Room 1 at either 9 AM or 10 AM.

**Step 2: Handle Constraint 1 (A before C)**
→ A must be at 9 AM, and C must be at 10 AM.

**Step 3: Handle Constraint 4 (A and B at the same time)**
→ Since A is at 9 AM, B must also be at 9 AM.

**Step 4: Handle Constraint 2 (B and D not in the same room)**
→ B is at 9 AM. D must be placed in a different room than B.

**Step 5: Place E**
→ Room 1 is available at either slot. Since A and B occupy two rooms at 9 AM, E can fit at 9 AM in the third room, or at 10 AM in Room 1.
→ Place **E at 10 AM in Room 1** (clean and avoids conflicts).

---

## Final Schedule

| Time Slot | Room 1 | Room 2 | Room 3 |
|-----------|--------|--------|--------|
| **9–10 AM** | A | B | D |
| **10–11 AM** | E | C | *(empty)* |

---

## Constraint Verification

| # | Constraint | Status |
|---|-----------|--------|
| 1 | **A before C** | ✅ A is at 9 AM, C is at 10 AM |
| 2 | **B and D not in same room** | ✅ B is in Room 2, D is in Room 3 |
| 3 | **E requires Room 1 (projector)** | ✅ E is scheduled in Room 1 |
| 4 | **A and B at the same time** | ✅ Both at 9–10 AM |

---

## Summary

- **Room 1:** A (9 AM) → E (10 AM)
- **Room 2:** B (9 AM) → C (10 AM)
- **Room 3:** D (9 AM)

All 5 meetings are scheduled within the 9–11 AM window across 3 rooms with **all constraints satisfied**. ✅

</details>

### Raw Extracted Constraints
```
1. When scheduling meetings with ordering constraints (A before C), enforce assigning the earlier meeting to the first available time slot and the later meeting to a subsequent slot; avoid placing them in the same time slot or reversing their order.

2. When a resource constraint restricts a meeting to a specific room (E requires Room 1), enforce placing that meeting in the designated room first before assigning other meetings; avoid scheduling conflicting meetings in that room without first resolving the resource-constrained meeting's placement.

3. When two meetings must occur simultaneously (A and B at the same time), enforce deriving their shared time slot from any other ordering constraints on either meeting before assigning rooms; avoid treating their time placement as independent decisions.

4. When verifying a proposed schedule, enforce checking every constraint explicitly against the final assignment rather than assuming correctness from the construction process; avoid declaring a schedule valid without a complete constraint-by-constraint verification pass.

5. When multiple meetings compete for a limited resource slot, enforce resolving conflicts by reassigning unconstrained meetings to alternative slots before removing already-placed meetings; avoid creating scheduling gaps or pushing meetings outside the allowed time window.

6. When constructing a schedule step-by-step, enforce propagating each assignment's implications to remaining unscheduled meetings immediately; avoid treating each meeting's placement as isolated from previously established constraints.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
