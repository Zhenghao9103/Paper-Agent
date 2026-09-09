import re

from fastapi.testclient import TestClient


def test_web_index_returns_frontend(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "PaperMind Agent" in response.text
    assert "pptx" not in response.text.lower()
    assert 'accept=".pdf"' in response.text
    assert 'accept=".pdf" multiple type="file"' in response.text
    assert 'id="upload-file-list"' in response.text
    assert "const maxUploadFiles = 18" in response.text
    assert "const uploadConcurrency = 2" in response.text
    assert "uploadSelectedFiles" in response.text
    assert "uploadOneFile" in response.text
    assert "Promise.all" in response.text
    assert "research-workbench" in response.text
    assert "--paper" in response.text
    assert "上传并解析" in response.text
    assert "知识库" in response.text
    assert "测试模型" in response.text
    assert "清空全部数据" in response.text
    assert "/api/documents/clear" in response.text
    assert "data-view-button=\"chat\"" in response.text
    assert "class=\"chat-shell\"" in response.text
    assert "id=\"chat-messages\"" in response.text
    assert "class=\"chat-composer\"" in response.text
    assert "appendChatMessage" in response.text
    assert "body: JSON.stringify({ question, session_id: currentSessionId })" in response.text
    assert "id=\"document-pagination\"" in response.text
    assert "id=\"document-page-status\"" in response.text
    assert "changeDocumentPage" in response.text
    assert "renderDocumentPage" in response.text
    assert "reparseDocument" in response.text
    assert "/reparse" in response.text
    assert "inspectDocument" in response.text
    assert "/elements" in response.text
    assert "/chunks" in response.text
    assert 'id="upload-progress"' in response.text
    assert 'id="upload-progress-label"' in response.text
    assert "xhr.upload.onprogress" in response.text
    assert "正在解析并建立索引" in response.text
    assert "上传或解析失败" in response.text
    assert "/api/documents/upload-async" in response.text
    assert "pollDocumentStatus" in response.text
    assert "cell_bboxes" not in response.text
    assert "暂无结构化描述" in response.text
    assert "result.citations" not in response.text
    assert "result.trace" not in response.text
    assert "sidebar-note" not in response.text
    assert "data-view-button=\"memory\"" not in response.text
    assert "id=\"pages\"" not in response.text
    assert "onclick=\"loadPages" not in response.text
    assert 'href="/static/chat/chat-runtime.css?v=20260908"' in response.text
    assert 'src="/static/chat/chat-runtime.js?v=20260908"' in response.text
    assert 'fetch("/api/chat/stream"' in response.text
    assert "PaperMindChat.consumeSse" in response.text
    assert "PaperMindChat.renderAnswer" in response.text
    assert "data-chat-status" in response.text
    assert "retryChatQuestion" in response.text
    assert 'fetch("/api/chat/ask"' not in response.text


def test_chat_runtime_assets_are_served_locally(client: TestClient) -> None:
    for path in (
        "/static/chat/chat-runtime.js",
        "/static/chat/chat-runtime.css",
        "/static/chat/fonts/KaTeX_Main-Regular.woff2",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.content


def test_web_index_has_no_remote_script_or_stylesheet_resources(client: TestClient) -> None:
    html = client.get("/").text
    resource_urls = re.findall(
        r'<(?:script[^>]+src|link[^>]+href)=["\']([^"\']+)',
        html,
        flags=re.IGNORECASE,
    )
    assert all(not url.startswith(("http://", "https://")) for url in resource_urls)
