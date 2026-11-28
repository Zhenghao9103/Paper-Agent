# Paper-Agent 3.0

<p align="center"><strong>“这篇论文的结论，哪组实验能支撑？”</strong></p>

<p align="center">
  上传 PDF，用中文提问，沿着引用核对原文。<br>
  你的研究兴趣和讲解偏好，下次对话也记得。
</p>

<p align="center">
  <a href="#快速开始"><strong>快速开始 ↓</strong></a>　·　
  <a href="frontend/public/readme/paper-agent-3-architecture.png">查看架构原图 ↗</a>
</p>

![Paper-Agent 3.0 架构图：论文知识库、研究问答与跨会话记忆](frontend/public/readme/paper-agent-3-architecture.png)

## 快速开始

准备 **64 位 Windows、Python 3.11 和 Git**。选择一种安装方式，在 PowerShell 中运行。

<details>
<summary><strong>本地路由</strong>　·　默认方式，需要 Git LFS</summary>

首次安装会下载约 1.1 GB 的路由模型。

```powershell
git lfs install
git clone https://github.com/Zhenghao9103/Paper-Agent.git
cd Paper-Agent
.\setup.ps1
```

</details>

<details>
<summary><strong>Jev 云端路由</strong>　·　使用 Jev API 密钥</summary>

此方式跳过本地路由模型下载。论文解析和检索仍在本地运行。

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/Zhenghao9103/Paper-Agent.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd Paper-Agent
.\setup.ps1 -RouterProvider jev
```

安装后，在项目根目录的 `.env` 中填写 `JEV_API_KEY`。

</details>

### 配置回答模型

编辑项目根目录的 `.env`，填入你的 OpenAI 兼容模型服务信息。两种安装方式都需要这一步。

```dotenv
AGENT_BASE_URL=https://your-provider.example/v1
AGENT_API_KEY=your_api_key
AGENT_MODEL=your_model
```

### 打开聊天页

```powershell
.\.venv\Scripts\python.exe main.py
```

浏览器会自动打开[本地聊天页](http://127.0.0.1:8000)。上传第一篇论文，等状态变为 `indexed`，就可以开始提问。
