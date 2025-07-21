# ✅ Instructions for Development
## 🔐 1. Do Not Modify LLM or Embedding Setup Code
The initialization logic for:
- `access_token`
- `embedding_model`
- `llm`

is defined once using organization - or environment-specific credentials and configurations. **Do not override or recreate these objects in other parts of the code.**

---

## 📌 2. Use These Global Variables Only
Wherever you need to use the LLM or embedding model in your logic, **always use these pre-initialized global variables**:

```python
access_token      # For token-authenticated environments (Azure or placeholder)
embedding_model   # LangChain-compatible embedding model instance
llm               # LangChain-compatible LLM instance
```

These are already declared and initialized in the startup phase of the application using:

```python
access_token = get_access_token()
embedding_model = get_embedding_model(access_token)
llm = get_llm(access_token)
```

> ✅ This allows seamless switching between local (OpenAI key-based) and org (Azure AD token-based) environments just by changing the functions `get_access_token()`, `get_llm()`, and `get_embedding_model()`.

---

## 🚫 3. Avoid Inline Instantiation
Do **not** write inline code like:

```python
ChatOpenAI(...)
OpenAIEmbeddings(...)
AzureChatOpenAI(...)
AzureOpenAIEmbeddings(...)
```

Instead, **use the variables**:

```python
response = llm.invoke(...)
embedding = embedding_model.embed_documents(...)
```

---