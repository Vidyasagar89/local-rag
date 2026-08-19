Prompt templates for conversation-aware Local RAG

1) Question rewrite (pre-retrieval)
Prompt:
"You are a concise query rewriter. Given the last 2 turns of the conversation and the user's new message, rewrite the new message into a standalone question preserving intent. Return ONLY the rewritten question, no commentary.

CONTEXT:
[RECENT_HISTORY]

USER MESSAGE:
[USER_MESSAGE]

OUTPUT:" 

Example: If RECENT_HISTORY contains: User: "What was my joining bonus?" Assistant: "Your joining bonus was Rs. 50,000, as stated in the offer letter." and USER_MESSAGE = "What is the condition?" then OUTPUT -> "What condition is the offer letter referring to about the joining bonus?"

2) Retriever input specification
- Use rewritten_query (from step 1) as the primary retrieval query.
- Also pass a short conversation_summary (1-2 sentences) and the last 2 turns as metadata / context to the retriever when supported.
- Retrieval params: top_k: 10 (was 8), use_metadata_bias: true if available.

3) Answer-generation prompt (grounded completion)
Prompt:
"System: You are an assistant that answers user questions using provided document snippets. Use the RECENT_HISTORY and the retrieved SNIPPETS. Always cite sources inline and at the end as a 'Sources:' list. If snippets don't answer the question, say: 'I could not find a specific definition in the indexed documents; here is an informed summary and next steps to verify.'

Input fields:
- RECENT_HISTORY: last 2 turns
- QUESTION: rewritten standalone question
- SNIPPETS: ranked retrieved passages with source identifiers

Response format requirements:
1. Short direct answer (1-3 sentences)
2. Quoted excerpt if it directly answers the question
3. Sources: list filenames and page/paragraph ids
4. If unsure, explicit "I could not find..." statement

4) Conversation summary (server-side optional)
- Instruction to model: produce a 1-2 sentence condensate of last N turns when requested: "Condense conversation to 1–2 sentences focusing on the user's intent and key facts." Cache and pass as conversation_summary.

5) Frontend payload recommendation
- Send: last_3_turns (user+assistant), allow_server_summary: true
- Server may ignore full history if conversation_summary is supplied

6) Chunking recommendation
- chunk_size: 500–800 tokens
- chunk_overlap: 150–200 tokens
- Ensure paragraph boundaries preserved when possible

7) Validation test case
- Example dialog and expected rewrite, retrieved excerpt, and answer citation for the "offer letter" scenario.
