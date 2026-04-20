import json
import sys

from flask import Flask, render_template, request

from sem_search_service import SemSearchService

app = Flask(__name__)
service = SemSearchService()


def _extract_batch_text(item: dict) -> str | None:
    text = item.get("data")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Backward compatibility for JSON files generated from PowerShell
    # Get-Content + ConvertTo-Json, where string values may be wrapped
    # as {"value": "...", ...}.
    if isinstance(text, dict):
        nested_value = text.get("value")
        if isinstance(nested_value, str) and nested_value.strip():
            return nested_value.strip()

    return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **service.build_page_context())


@app.route("/add", methods=["POST"])
def add_view():
    context = service.build_page_context()
    raw_text = request.form.get("add_text", "").strip()
    upload = request.files.get("add_file")
    context["add_text"] = raw_text

    if upload and upload.filename:
        try:
            payload = json.load(upload.stream)
        except json.JSONDecodeError:
            context["add_message"] = "JSON 文件解析失败，请使用有效的 JSON 格式。"
            return render_template("index.html", **context), 400

        if not isinstance(payload, list):
            context["add_message"] = "JSON 文件内容必须是数组，例如：[{\"data\": \"文本内容\"}]。"
            return render_template("index.html", **context), 400

        texts = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                context["add_message"] = f"第 {index} 条数据不是对象，正确格式应为 {{\"data\": \"文本内容\"}}。"
                return render_template("index.html", **context), 400

            text = _extract_batch_text(item)
            if text is None:
                context["add_message"] = f"第 {index} 条数据缺少非空字符串字段 data。"
                return render_template("index.html", **context), 400

            texts.append(text)
    else:
        if not raw_text:
            context["add_message"] = "请输入要写入向量库的内容，或上传 JSON 批量导入文件。"
            return render_template("index.html", **context), 400

        texts = [raw_text]

    insert_count = service.add_contents(texts)
    context["add_message"] = f"写入成功，共插入 {insert_count} 条文本。"
    context["add_text"] = ""
    return render_template("index.html", **context)


@app.route("/search", methods=["POST"])
def search_view():
    context = service.build_page_context()
    query = request.form.get("search_query", "").strip()
    top_k_raw = request.form.get("top_k", "5").strip()

    context["search_query"] = query
    context["top_k"] = top_k_raw or "5"

    if not query:
        context["search_message"] = "请输入要检索的内容。"
        return render_template("index.html", **context), 400

    try:
        top_k = max(1, min(int(top_k_raw), 20))
    except ValueError:
        context["search_message"] = "TopN 必须是 1 到 20 之间的整数。"
        return render_template("index.html", **context), 400

    context["top_k"] = top_k
    results = service.search_contents(query, top_k)
    context["search_results"] = results
    context["search_message"] = (
        f"检索完成，返回 {len(results)} 条结果。"
        if results
        else "未检索到相似内容。"
    )
    return render_template("index.html", **context)


@app.route("/health", methods=["GET"])
def health():
    return service.health_payload()


if __name__ == "__main__":
    debug_mode = True
    # PyCharm debugger already manages the process lifecycle. Disable Flask's
    # stat reloader there to avoid restarting through a broken helper path.
    use_reloader = sys.gettrace() is None
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=use_reloader)
