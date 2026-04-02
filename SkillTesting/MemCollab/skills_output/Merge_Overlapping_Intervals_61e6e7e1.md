---
name: Merge Overlapping Intervals
domain: Coding / Algorithms / Array Processing
generated: 2026-04-02 09:03 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Merge Overlapping Intervals

> How to correctly merge all overlapping intervals in a list, handling unsorted input, swallowed intervals, adjacent intervals, and edge cases without mutating the caller's data.

## When to Apply

Apply whenever you need to consolidate a collection of potentially overlapping, adjacent, or unsorted ranges/intervals into a minimal set of non-overlapping intervals — common in scheduling, calendar merging, genomics range queries, and geometric sweep problems.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Always sort intervals by start value before iterating; overlaps between non-adjacent pairs are invisible in unsorted order.
- Two intervals overlap (or are adjacent) if and only if current_start <= last_merged_end; strict less-than incorrectly splits touching intervals.
- When merging, update the running end with max(last_end, current_end) to correctly handle the case where a smaller interval is fully contained inside a larger one.
- Never mutate the original input; produce a new sorted list and copy intervals before appending to the result.
- Handle empty and single-element inputs explicitly before entering the main loop to avoid aliasing and off-by-one bugs.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Processing intervals in their original unsorted order — overlaps between non-adjacent pairs will be silently missed.
- ⚠️ Using strict less-than (current_start < last_end) to detect overlap — adjacent intervals like [1,3],[3,5] are incorrectly left unmerged.
- ⚠️ Assigning current_end directly instead of max(last_end, current_end) — a contained interval like [2,5] inside [1,10] would incorrectly shrink the merged result.
- ⚠️ Aliasing intervals from the input directly into the result list — later mutations to the result corrupt the caller's original data.
- ⚠️ Relying on the general loop to implicitly handle empty or single-element input — subtle aliasing or index errors can occur.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce sorting with sorted(intervals, key=lambda x: x[0]) to produce a new list; avoid in-place .sort() which silently mutates the caller's input.
- Enforce overlap check using current_start <= last_end (<=, not <); avoid strict < which treats adjacent intervals as non-overlapping.
- Enforce end-update via max(last_end, current_end) when merging; avoid direct assignment of current_end which breaks the swallow case.
- Enforce appending copies of intervals (e.g., interval.copy() or list slicing) to the result; avoid direct aliasing that creates shared mutable state.
- Enforce explicit early-return guards for empty list and single-element list before the main loop; avoid assuming the loop handles these safely.

---

## Example Pattern

```
INPUT: [[8,10],[1,3],[2,6],[15,18]] → SORT: [[1,3],[2,6],[8,10],[15,18]] → ITERATE: start result with copy of first interval; for each subsequent interval, if current_start <= result[-1][1] then result[-1][1] = max(result[-1][1], current_end), else append copy of current interval → OUTPUT: [[1,6],[8,10],[15,18]]
```

---

## Source: Contrastive Trajectory Analysis

### Task
```
Write a Python function to merge all overlapping intervals in a list of intervals (e.g., [[1,3],[2,6],[8,10],[15,18]]). The input might be unsorted. Explain your logic and how you handle edge cases where one interval completely swallows another.
```

### Weak Agent Trajectory (`meta-llama/llama-3.1-8b-instruct`)
<details>
<summary>Expand weak agent reasoning</summary>

**Merging Overlapping Intervals in Python**
=====================================================

### Problem Statement

Given a list of intervals, merge all overlapping intervals into a new list of non-overlapping intervals.

### Logic

To solve this problem, we will use the following steps:

1. Sort the list of intervals based on the start value of each interval.
2. Initialize an empty list to store the merged intervals.
3. Iterate through the sorted list of intervals. For each interval, check if it overlaps with the last merged interval in the result list. If it does, merge the two intervals by updating the end value of the last merged interval. If it doesn't, append the current interval to the result list.
4. Return the list of merged intervals.

### Python Implementation

```python
def merge_intervals(intervals):
    # Sort the intervals based on the start value of each interval
    intervals.sort(key=lambda x: x[0])

    merged = []
    for current in intervals:
        # Check if the result list is empty or the current interval does not overlap with the last merged interval
        if not merged or current[0] > merged[-1][1]:
            merged.append(current)
        else:
            # Merge the current interval with the last merged interval
            merged[-1][1] = max(merged[-1][1], current[1])

    return merged
```

### Example Use Cases

```python
print(merge_intervals([[1,3],[2,6],[8,10],[15,18]))  # Output: [[1,6],[8,10],[15,18]]
print(merge_intervals([[1,4],[4,5]])  # Output: [[1,5]]
print(merge_intervals([[1,3],[2,3]])  # Output: [[1,3]]
print(merge_intervals([[1,6],[2,3],[8,10],[15,18]])  # Output: [[1,6],[8,10],[15,18]]
```

### Handling Edge Cases

The function handles the following edge cases:

*   **Empty list of intervals:** If the input list is empty, the function returns an empty list.
*   **Single interval:** If the input list contains only one interval, the function returns a list containing that interval.
*   **Non-overlapping intervals:** If the input list contains non-overlapping intervals, the function returns a list of non-overlapping intervals.
*   **One interval completely swallows another:** If one interval completely swallows another, the function merges the two intervals into a single interval.

Note that the function assumes that the input intervals are valid (i.e., the start value is less than or equal to the end value). If the input intervals are not valid, the function may produce incorrect results.

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

# Merging Overlapping Intervals in Python

## Logic & Approach

### Step-by-Step Strategy:
1. **Sort** intervals by their start value
2. **Iterate** through sorted intervals, comparing each with the last merged interval
3. **Merge** if the current interval's start ≤ last merged interval's end (overlap exists)
4. **Append** as-is if no overlap

### Edge Cases Handled:
| Case | Example | Behavior |
|------|---------|----------|
| Empty input | `[]` | Return `[]` |
| Single interval | `[[5,10]]` | Return `[[5,10]]` |
| Complete swallow | `[[1,10],[2,5]]` | `[1,10]` absorbs `[2,5]` |
| Adjacent intervals | `[[1,3],[3,5]]` | Merged → `[[1,5]]` |
| No overlaps | `[[1,2],[3,4]]` | Unchanged |
| Unsorted input | `[[8,10],[1,3]]` | Sorted first, then merged |
| Duplicate intervals | `[[2,4],[2,4]]` | Returns `[[2,4]]` |

---

## Full Implementation

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merges all overlapping intervals in a list.

    Args:
        intervals: A list of [start, end] pairs (may be unsorted).

    Returns:
        A sorted list of merged non-overlapping intervals.

    Examples:
        >>> merge_intervals([[1,3],[2,6],[8,10],[15,18]])
        [[1, 6], [8, 10], [15, 18]]
        >>> merge_intervals([[1,4],[4,5]])
        [[1, 5]]
        >>> merge_intervals([[1,10],[2,5]])   # swallow case
        [[1, 10]]
    """

    # ── Edge case: empty or single-element list ──────────────────────────────
    if len(intervals) <= 1:
        return intervals.copy()  # return a copy to avoid mutating the input

    # ── Step 1: Sort by start value (stable sort keeps equal starts ordered) ─
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # ── Step 2: Initialize merged list with the first interval ───────────────
    merged = [sorted_intervals[0].copy()]  # copy to avoid aliasing

    # ── Step 3: Iterate through remaining intervals ──────────────────────────
    for current_start, current_end in sorted_intervals[1:]:

        last_start, last_end = merged[-1]  # peek at the last merged interval

        # ── Overlap condition: current starts before or when last ends ───────
        # Also handles "swallow": if current_end <= last_end, max keeps last_end
        if current_start <= last_end:
            # Extend the last interval's end if current reaches further
            merged[-1][1] = max(last_end, current_end)
            #                   ↑ KEY: max() handles the "swallow" case
            #   e.g., last=[1,10], current=[2,5] → max(10,5)=10 → [1,10] ✓
            #   e.g., last=[1,5],  current=[3,8] → max(5,8)=8  → [1,8]  ✓

        # ── No overlap: add current interval as a new entry ──────────────────
        else:
            merged.append([current_start, current_end])

    return merged


# ── Demonstration ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # (description, input)
        ("Standard case",         [[1,3],[2,6],[8,10],[15,18]]),
        ("Unsorted input",        [[8,10],[1,3],[2,6],[15,18]]),
        ("Swallow (inner eaten)", [[1,10],[2,5],[3,7]]),
        ("Adjacent intervals",    [[1,3],[3,5],[5,7]]),
        ("No overlaps",           [[1,2],[3,4],[5,6]]),
        ("Single interval",       [[5,10]]),
        ("Empty list",            []),
        ("Duplicates",            [[2,4],[2,4],[1,3]]),
        ("All same",              [[1,5],[1,5],[1,5]]),
    ]

    for description, intervals in test_cases:
        result = merge_intervals(intervals)
        print(f"{description:<25} {str(intervals):<35} → {result}")
```

---

## Output

```
Standard case             [[1, 3], [2, 6], [8, 10], [15, 18]]    → [[1, 6], [8, 10], [15, 18]]
Unsorted input            [[8, 10], [1, 3], [2, 6], [15, 18]]    → [[1, 6], [8, 10], [15, 18]]
Swallow (inner eaten)     [[1, 10], [2, 5], [3, 7]]              → [[1, 10]]
Adjacent intervals        [[1, 3], [3, 5], [5, 7]]               → [[1, 7]]
No overlaps               [[1, 2], [3, 4], [5, 6]]               → [[1, 2], [3, 4], [5, 6]]
Single interval           [[5, 10]]                              → [[5, 10]]
Empty list                []                                      → []
Duplicates                [[2, 4], [2, 4], [1, 3]]               → [[1, 4]]
All same                  [[1, 5], [1, 5], [1, 5]]               → [[1, 5]]
```

---

## Visual Walkthrough

```
Input: [[1,3],[2,6],[8,10],[15,18]]

After sort: [[1,3],[2,6],[8,10],[15,18]]  (already sorted)

Step 1: merged = [[1,3]]
        ├─ current=[2,6]: 2 <= 3? YES → merge → [1, max(3,6)] = [1,6]
        │   merged = [[1,6]]

Step 2: current=[8,10]: 8 <= 6? NO  → append
        │   merged = [[1,6],[8,10]]

Step 3: current=[15,18]: 15 <= 10? NO → append
            merged = [[1,6],[8,10],[15,18]]

Result: [[1,6],[8,10],[15,18]] ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Swallow case: [[1,10],[2,5],[3,7]]

After sort: [[1,10],[2,5],[3,7]]

Step 1: merged = [[1,10]]
        ├─ current=[2,5]: 2 <= 10? YES → [1, max(10,5)] = [1,10]  ← swallow!
        ├─ current=[3,7]: 3 <= 10? YES → [1, max(10,7)] = [1,10]  ← swallow!

Result: [[1,10]] ✓
```

---

## Complexity Analysis

| Metric | Value | Reason |
|--------|-------|--------|
| **Time** | O(n log n) | Dominated by the sort step |
| **Space** | O(n) | Output list in worst case (no overlaps) |
| **In-place?** | No | Input is not mutated (safe copies made) |

The **key insight** is `max(last_end, current_end)` — after sorting by start, we only need to check if the current interval *starts* before the last one *ends*. If it does, we extend the end to whichever is larger, naturally handling the "swallow" case where a small interval is completely inside a larger one.

</details>

### Raw Extracted Constraints
```
1. When merging overlapping intervals, enforce sorting by start value before iterating; avoid processing unsorted intervals as overlaps may be missed between non-adjacent pairs.

2. When checking if two intervals overlap after sorting, enforce using `current_start <= last_end` (not strict `<`); avoid using strict less-than which incorrectly treats adjacent intervals like `[1,3],[3,5]` as non-overlapping.

3. When merging an overlapping interval into the last merged interval, enforce updating the end with `max(last_end, current_end)`; avoid simply assigning `current_end` which fails the "swallow" case where a smaller interval is contained within a larger one.

4. When implementing interval merging, enforce creating copies of intervals (via `.copy()` or equivalent) to avoid mutating the original input list; avoid direct aliasing that causes unexpected side effects on the caller's data.

5. When handling edge cases for interval merging, enforce explicit checks for empty or single-element input before the main logic; avoid relying on the general loop to handle these cases as it may introduce off-by-one or aliasing bugs.

6. When sorting intervals as a preprocessing step, enforce using a new sorted list (`sorted(...)`) rather than in-place `.sort()`; avoid in-place sorting when input immutability is required, as it silently mutates the caller's data.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
