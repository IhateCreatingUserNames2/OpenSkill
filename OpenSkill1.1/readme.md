# The Universal Skill Crafter

This is a Proof of Concept Prototype. 

## Overview
The **Universal Skill Crafter** is an automated engine designed to extract, refine, and store "Universal Skills" from AI reasoning trajectories. By utilizing a heterogeneous multi-model approach, it distills high-level logic into an agent-agnostic format, ensuring that knowledge can be shared across diverse LLM architectures without "style contamination."

---

## Core Methodologies

### 1. MemCollab (Contrastive Distillation)
The system runs a **Weak Agent** and a **Strong Agent** on the same task. It identifies the "Success Path" and the "Failure Path" to extract:
* **Reasoning Invariants**: Logical principles that must be followed.
* **Violation Patterns**: Forbidden patterns that lead to failure.



### 2. Trace2Skill (Skill Evolution)
Every reasoning trace is converted into a structured `.md` file. These skills are not static; the system tracks an `evolution_count`, allowing the skill to be updated and refined as more trajectories are analyzed.

### 3. TurbOQuant (Knowledge Quantization)
To handle massive skill banks efficiently, the system uses **TurbOQuant**. It quantizes high-dimensional embeddings into low-bit representations.
* **Residual Quantization**: It stores a base vector (`qvec`) and the error/difference (`residual`).
* **Efficiency**: Uses 4-bit quantization and scaling to maintain accuracy while drastically reducing memory footprint.

### 4. S-PATH RAG (Path-Aware Retrieval)
Skills are stored in a **Skill Graph**. S-PATH RAG allows the system to navigate this graph based on intent and category, ensuring that the model retrieves the exact "logical handbook" needed for a specific subcategory.

---

## How to Use
1.  **Define Task**: Input a complex problem into the script.
2.  **Trajectory Generation**: The script queries the Weak and Strong LLMs.
3.  **Distillation**: MemCollab extracts the logic.
4.  **Vectorization**: TurbOQuant compresses the knowledge.
5.  **Graph Update**: The new skill is added to the `skill_graph.json`.

---

## Technical Analysis of Results

### 1. The Skill Metadata (JSON)
* **Compression Efficiency**: The `qvector` implementation of **TurbOQuant** successfully utilizes **4-bit** quantization and **Residuals**. This confirms the system can store complex logic with minimal storage overhead.
* **Heterogeneous Pairing**: The JSON identifies the **gpt-oss-120b** (Weak) and **minimax-m2.7** (Strong) pairing, which is the prerequisite for MemCollab contrastive analysis.

### 2. The Skill MD (`Raft-Traffic-Optimized...md`)
* **Normative Constraints**: The generated skill includes the "Enforce/Avoid" format. For example, it enforces checking the `lastLogIndex` before a heart-beat—a classic **Reasoning Invariant** for the Raft protocol.
* **Violation Patterns**: It successfully identifies "blindly increasing the commitIndex" as a failure mode, preventing smaller models (e.g., 7B) from hallucinating a consensus.

### 3. The Skill Graph JSON
* **Topological Foundation**: Currently initialized with a single node (`a493a4b2`).
* **S-PATH Potential**: The `edges` list is structured for **S-PATH RAG** to connect this skill to related concepts (e.g., "Paxos Consensus" or "Gossip Protocols") based on `subcategory` and `domain`.

---

## Conclusion
The system successfully creates **Quantized Universal Skills**. Rather than simple "how-to" guides, it saves **compressed logical frameworks** indexed in a Knowledge Graph, ready for retrieval by any model seeking to optimize distributed systems logic.

## References
* **S-Path-RAG**: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering https://arxiv.org/abs/2603.23512 
* **TurboQuant**: Online Vector Quantization with Near-optimal Distortion Rate https://arxiv.org/abs/2504.19874 
* **MemCollab**: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation https://arxiv.org/abs/2603.23234 
* **Trace2Skill**: Distill Trajectory-Local Lessons into Transferable Agent Skills https://arxiv.org/abs/2603.25158 
