def normalize_messages(messages):

    normalized = []

    for m in messages:

        role = getattr(m, "type", None)

        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role not in ["system", "user", "assistant"]:
            role = "user"

        content = getattr(m, "content", "")

        normalized.append({
            "role": role,
            "content": str(content)
        })

    return normalized