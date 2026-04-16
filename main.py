from flask import Flask, render_template, request

from sem_search_service import SemSearchService

app = Flask(__name__)
service = SemSearchService()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **service.build_page_context())


@app.route("/add", methods=["POST"])
def add_view():
    context = service.build_page_context()
    raw_text = request.form.get("add_text", "").strip()
    context["add_text"] = raw_text

    if not raw_text:
        context["add_message"] = "请输入要写入向量库的内容。"
        return render_template("index.html", **context), 400

    texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
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
    app.run(host="0.0.0.0", port=5000, debug=True)
