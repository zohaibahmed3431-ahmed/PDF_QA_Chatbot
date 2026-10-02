def format_sources(results: list[dict]) -> str:
    if not results:
        return ""

    lines = ["**📚 Sources**"]
    seen = set()

    for item in results[:6]:
        source = item.get("source", "Unknown")
        page = item.get("page")
        sheet = item.get("sheet")

        if page is not None:
            location = f"Page {page}"
        elif sheet:
            location = f"Sheet: {sheet}"
        else:
            location = "Document content"

        key = (source, location)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- 📄 `{source}` — {location}")

    return "\n".join(lines)
