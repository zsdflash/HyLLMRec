# 📘 From Text to Extra Semantics: Leveraging Large Language Models for Graph-Based Multi-Modal Recommendation

> **HyLLMRec** — Official (research) repository for the paper  
> **From Text to Extra Semantics: Leveraging Large Language Models for Graph-Based Multi-Modal Recommendation**  
> *Shaodong Zhang*, *Bingcai Chen* (Dalian University of Technology, China)

---

## 🔍 Overview

**HyLLMRec** is a **hypergraph-based, LLM-enhanced multi-modal recommendation framework**.  
It targets core challenges in multi-modal recommendation where **textual and visual signals are sparse, short, and noisy**.  
HyLLMRec enriches item semantics with **LLM-driven keyword extraction and semantic association**, and **induces learnable hyperedges** to model **high-order user–item–feature relations**.  
A **visual–textual cross-attention** module aligns modalities, and a **tri-view representation** (collaborative, modal, hypergraph) is learned with a unified objective.
---

## 🚀 Key Features

- **LLM-Driven Semantic Expansion**  
  Extracts salient **keywords** and generates **complementary semantic attributes** from raw item text to combat sparsity and noise.

- **Learnable Hypergraph Construction**  
  Uses **Gumbel–Softmax** and **multi-source fusion** to **adaptively induce hyperedges**, capturing high-order semantics beyond pairwise graphs.

- **Visual–Textual Cross-Attention**  
  Aligns vision and language features so that visual cues gain semantic grounding from text, and vice versa.

- **Tri-View Representation Learning**  
  Integrates **collaborative**, **modality-guided**, and **semantic-hypergraph** views with a unified optimization objective.

---

## 📚 Datasets (used in the paper)

Public Amazon subsets with high sparsity:

- **Baby**  
- **Office**  
- **Game**
