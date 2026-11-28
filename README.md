# Paper-Agent 3.0

上传论文 PDF，用中文提问，沿着回答中的引用核对原文。Paper-Agent 还能记住你的研究兴趣，以及你希望它怎样讲解问题。

![Paper-Agent 3.0 架构图：论文知识库、研究问答与跨会话记忆](frontend/public/readme/paper-agent-3-architecture.png)

## 快速开始

准备一台 **64 位 Windows** 电脑，安装 **Python 3.11、Git 和 Git LFS**，并在安装盘预留至少 **20 GB** 空间。首次安装需要联网下载模型。

**1. 安装。** 打开 PowerShell，运行以下命令。

```powershell
git lfs install
git clone https://github.com/Zhenghao9103/Paper-Agent.git
cd Paper-Agent
.\setup.ps1
```

**2. 配置模型。** 在项目根目录的 `.env` 文件中填入模型服务商提供的 API 地址、密钥和模型名。支持 OpenAI 兼容接口。

```dotenv
AGENT_BASE_URL=https://your-provider.example/v1
AGENT_API_KEY=your_api_key
AGENT_MODEL=your_model
```

**3. 开始读论文。** 在同一目录运行启动命令，浏览器会打开[聊天页](http://127.0.0.1:8000)。

```powershell
.\.venv\Scripts\python.exe main.py
```

上传一篇 PDF，等文档状态显示 `indexed` 后，试着问：

> 这篇论文最重要的结论是什么？哪组实验支持它？请标出原文出处。

<details>
<summary>从 2.2 升级</summary>

先停止服务，备份 SQLite 数据库和 Chroma 目录。更新代码后，在项目根目录运行以下命令，再重新启动。

```powershell
.\.venv\Scripts\python.exe -m alembic -c backend/alembic.ini upgrade head
```

</details>
