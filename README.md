"# KRMBot" 
 Architecture du Projet
┌─────────────────────────────────────────────────────────┐
│                  PIPELINE RAG                           │
│                                                         │
│  📄 Documents  ──►  ✂️ Chunking  ──►  🔢 Embeddings    │
│                                            │            │
│                                            ▼            │
│  💬 Question  ──►  🔢 Embedding  ──►  🗄️ ChromaDB      │
│                                            │            │
│                                            ▼            │
│                         📋 Top-K Chunks pertinents      │
│                                            │            │
│                                            ▼            │
│                    🧠 LLM + Prompt enrichi              │
│                                            │            │
│                                            ▼            │
│                         💡 Réponse + Sources citées     │
└─────────────────────────────────────────────────────────┘
