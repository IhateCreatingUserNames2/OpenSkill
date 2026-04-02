---
name: Interval Merging
domain: Coding / Algorithms / Array Processing
generated: 2026-04-02 08:28 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Interval Merging

> How to correctly merge overlapping intervals in a list, including sorting, overlap detection, swallowed-interval handling, chain-merge correctness, and safe input handling.

## When to Apply

Apply whenever merging, collapsing, or deduplicating a list of intervals or ranges, including scheduling, calendar overlap resolution, genomic region merging, or any domain where contiguous or overlapping spans must be unified.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Always sort intervals by start value before any comparison or merge logic
- Use <= (not <) when checking overlap to correctly handle touching/adjacent intervals
- Use max(last_end, current_end) as the merged endpoint to handle swallowed intervals
- Always compare a new interval against the last element of the merged output list, not the previous input interval
- Work on copies of input data to avoid mutating the caller's data structures
- Explicitly handle empty and single-element inputs with early returns before entering the main loop

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Attempting to merge intervals without sorting first, producing incorrect or missed merges
- ⚠️ Using strict < for overlap check, incorrectly treating touching boundaries as non-overlapping
- ⚠️ Taking the current interval's end directly as the merged end, missing cases where the previous interval swallows the current one
- ⚠️ Comparing a new interval against the previous input interval instead of merged[-1], causing missed chain-merges
- ⚠️ Mutating input interval elements in-place, introducing hidden side effects for the caller
- ⚠️ Running main loop logic on empty or single-element inputs without guard clauses, risking index errors

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce sorting by start value (intervals.sort(key=lambda x: x[0])) before any merge logic; avoid processing intervals in arbitrary order
- Enforce overlap condition current_start <= last_merged_end using <=; avoid strict < which misclassifies adjacent intervals as non-overlapping
- Enforce merged endpoint as max(last_merged_end, current_end); avoid using current_end directly which drops larger enclosing intervals
- Enforce comparing each new interval against merged[-1] (last merged result); avoid comparing against the raw previous input interval to ensure chain-merges collapse correctly
- Enforce creating copies of intervals and output list elements; avoid in-place modification of input list or its sublists
- Enforce early return of [] for empty input and [copy of single interval] for single-element input; avoid entering loop logic on degenerate inputs

---

## Example Pattern

```
Given intervals [[1,3],[2,6],[8,10],[15,18]]: sort -> [[1,3],[2,6],[8,10],[15,18]]; initialize merged with copy of first -> [[1,3]]; check 2<=3 -> overlap -> update end to max(3,6)=6 -> [[1,6]]; check 8<=6 -> no overlap -> append -> [[1,6],[8,10]]; check 15<=10 -> no overlap -> append -> [[1,6],[8,10],[15,18]]; return result.
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

We will create a Python function `merge_intervals` that takes a list of intervals as input and returns a new list with merged intervals.

**Code**
--------

```python
def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list of intervals.

    Args:
        intervals (list): A list of intervals, where each interval is a list of two integers.

    Returns:
        list: A new list with merged intervals.
    """
    # Sort the intervals by their start value
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # If the list of merged intervals is empty or if the current interval does not overlap with the previous, append it
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, there is overlap, so we merge the current and previous intervals
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged
```

**Logic**
--------

1.  **Sorting**: First, we sort the input list of intervals by their start value. This is necessary for the following steps, as we will be iterating over the list and merging intervals in order of their start value.
2.  **Iterating over intervals**: We then iterate over each interval in the sorted list.
3.  **Checking for overlap**: For each interval, we check if the list of merged intervals is empty or if the current interval's start value is greater than the end value of the last merged interval. If either condition is true, we append the current interval to the `merged` list.
4.  **Merging intervals**: If there is an overlap, we merge the current and previous intervals by updating the end value of the last merged interval to the maximum of its current end value and the end value of the current interval.
5.  **Returning the merged intervals**: Finally, we return the list of merged intervals.

**Handling Edge Cases**
--------------------

*   **Empty input**: If the input list is empty, the function will return an empty list.
*   **Single interval**: If the input list contains only one interval, the function will return the same list.
*   **Non-overlapping intervals**: If the input list contains non-overlapping intervals, the function will return the original list.
*   **Completely swallowed intervals**: If one interval completely swallows another, the function will merge them correctly. For example, if the input is `[[1,3],[2,6],[8,10],[15,18]]`, the function will return `[[1,6],[8,10],[15,18]]`.

**Example Use Cases**
---------------------

```python
print(merge_intervals([[1,3],[2,6],[8,10],[15,18]])  # [[1,6],[8,10],[15,18]]
print(merge_intervals([[1,4],[4,5]])  # [[1,5]]
print(merge_intervals([])  # []
print(merge_intervals([[1,3]])  # [[1,3]]
```

**Commit Message**
---------------

`Added function to merge overlapping intervals in a list of intervals`

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

# Merging Overlapping Intervals in Python

## Logic & Approach

### Core Idea
1. **Sort** the intervals by their start value
2. **Iterate** through sorted intervals, comparing each with the last merged interval
3. **Merge** if the current interval overlaps with (or touches) the last merged one
4. **Append** if it doesn't overlap

### Overlap Condition
Two intervals `[a, b]` and `[c, d]` overlap when `c <= b` (after sorting, so `a <= c` is guaranteed).

### "Swallowing" Edge Case
If `[1, 10]` is followed by `[2, 5]`, then `5 <= 10`, so they overlap.
The merged result should be `[1, max(10, 5)] = [1, 10]` — the larger interval **swallows** the smaller one.
Using `max()` for the end value handles this automatically.

---

## Full Implementation

```python
def merge_intervals(intervals):
    """
    Merges all overlapping intervals in a list.

    Args:
        intervals: List of [start, end] pairs (may be unsorted, may contain edge cases)

    Returns:
        List of merged non-overlapping intervals, sorted by start value.

    Edge Cases Handled:
        - Empty list
        - Single interval
        - Completely swallowed intervals (e.g., [1,10] swallows [2,5])
        - Adjacent intervals (e.g., [1,3] and [3,5] → [1,5])
        - Identical intervals
        - Unsorted input
    """

    # ── Edge case: empty or single-element list ──────────────────────────────
    if len(intervals) <= 1:
        return intervals.copy()  # Return a copy to avoid mutating the original

    # ── Step 1: Sort by start value ──────────────────────────────────────────
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # ── Step 2: Initialize merged list with the first interval ───────────────
    merged = [sorted_intervals[0].copy()]  # Copy to avoid mutating input

    # ── Step 3: Iterate and merge ────────────────────────────────────────────
    for current in sorted_intervals[1:]:
        current_start, current_end = current
        last_merged = merged[-1]  # Always compare against the LAST merged interval

        if current_start <= last_merged[1]:
            # ── OVERLAP (or adjacent): Merge by extending the end ──────────
            # max() handles the "swallowing" case:
            #   last=[1,10], current=[2,5] → max(10,5)=10 → [1,10]  ✓
            #   last=[1,3],  current=[2,6] → max(3,6)=6  → [1,6]   ✓
            last_merged[1] = max(last_merged[1], current_end)
        else:
            # ── NO OVERLAP: Start a new interval ───────────────────────────
            merged.append(current.copy())

    return merged


# ════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ════════════════════════════════════════════════════════════════════════════

def run_tests():
    test_cases = [
        # (description, input, expected_output)
        (
            "Standard overlapping intervals (unsorted)",
            [[8, 10], [1, 3], [15, 18], [2, 6]],
            [[1, 6], [8, 10], [15, 18]]
        ),
        (
            "One interval completely swallows another",
            [[1, 10], [2, 5], [3, 7]],
            [[1, 10]]
        ),
        (
            "Adjacent intervals (touching boundaries)",
            [[1, 3], [3, 5], [5, 7]],
            [[1, 7]]
        ),
        (
            "No overlaps at all",
            [[1, 2], [4, 5], [7, 8]],
            [[1, 2], [4, 5], [7, 8]]
        ),
        (
            "All intervals identical",
            [[2, 4], [2, 4], [2, 4]],
            [[2, 4]]
        ),
        (
            "Single interval",
            [[5, 9]],
            [[5, 9]]
        ),
        (
            "Empty list",
            [],
            []
        ),
        (
            "Two non-overlapping intervals",
            [[1, 3], [5, 7]],
            [[1, 3], [5, 7]]
        ),
        (
            "Complex chain merge",
            [[1, 4], [2, 5], [4, 8], [7, 10], [12, 15]],
            [[1, 10], [12, 15]]
        ),
    ]

    all_passed = True
    for desc, input_data, expected in test_cases:
        result = merge_intervals(input_data)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result != expected:
            all_passed = False
        print(f"{status} | {desc}")
        if result != expected:
            print(f"       Input:    {input_data}")
            print(f"       Expected: {expected}")
            print(f"       Got:      {result}")

    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️  Some tests failed."))


run_tests()
```

---

## Output

```
✅ PASS | Standard overlapping intervals (unsorted)
✅ PASS | One interval completely swallows another
✅ PASS | Adjacent intervals (touching boundaries)
✅ PASS | No overlaps at all
✅ PASS | All intervals identical
✅ PASS | Single interval
✅ PASS | Empty list
✅ PASS | Two non-overlapping intervals
✅ PASS | Complex chain merge

🎉 All tests passed!
```

---

## Visual Walkthrough

```
Input: [[8,10],[1,3],[15,18],[2,6]]

After sorting: [[1,3],[2,6],[8,10],[15,18]]

Step 1: merged = [[1,3]]
        current = [2,6] → 2 <= 3? YES → merge → [1, max(3,6)] = [1,6]
        merged = [[1,6]]

Step 2: merged = [[1,6]]
        current = [8,10] → 8 <= 6? NO  → append
        merged = [[1,6],[8,10]]

Step 3: merged = [[1,6],[8,10]]
        current = [15,18] → 15 <= 10? NO → append
        merged = [[1,6],[8,10],[15,18]]

Final:  [[1,6],[8,10],[15,18]] ✓
```

---

## Complexity Analysis

| Aspect | Value | Reason |
|--------|-------|--------|
| **Time** | O(n log n) | Dominated by sorting |
| **Space** | O(n) | Output list in worst case (no overlaps) |

</details>

### Raw Extracted Constraints
```
1. When merging overlapping intervals, enforce sorting by start value first before any comparison logic; avoid attempting to merge intervals in unsorted order as it produces incorrect results.

2. When checking if two intervals overlap after sorting, enforce the condition `current_start <= last_merged_end` (using `<=` to handle adjacent/touching intervals); avoid using strict `<` which incorrectly treats touching boundaries as non-overlapping.

3. When merging two overlapping intervals, enforce using `max(last_end, current_end)` for the merged endpoint; avoid simply taking the current interval's end, as one interval may be completely swallowed by another with a larger endpoint.

4. When implementing interval merging, enforce making copies of input intervals and the output list to avoid mutating the original input data; avoid in-place modification of input elements which causes hidden side effects.

5. When handling edge cases for interval merging, enforce explicit checks for empty lists and single-element lists with early returns; avoid running the main loop logic on degenerate inputs that could cause index errors or incorrect behavior.

6. When always comparing a new interval against the merged result, enforce comparing only against the last element of the merged list (`merged[-1]`), not the previous input interval; avoid comparing against the wrong reference which would miss chain-merges like `[1,4],[2,5],[4,8]` collapsing into `[1,8]`.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
