/**
 * DeepSeek Code Agent — VS Code 扩展入口
 * 
 * 功能：
 * - 侧边栏对话窗口（Webview）
 * - 右键菜单：修复 / 审查 / 查文档
 * - SCM 菜单：智能提交
 * - 终端输出流式响应
 */

'use strict';

const vscode = require('vscode');
const { WebviewPanel } = vscode;

/** @type {vscode.ExtensionContext} */
let context;

/**
 * 获取配置
 */
function getConfig() {
    return vscode.workspace.getConfiguration('deepseek.agent');
}

function getApiKey() {
    return getConfig().get('apiKey') || process.env.DEEPSEEK_API_KEY || '';
}

/**
 * 激活扩展
 */
function activate(ctx) {
    context = ctx;

    // 注册命令
    ctx.subscriptions.push(
        vscode.commands.registerCommand('deepseek-agent.chat', openChatPanel),
        vscode.commands.registerCommand('deepseek-agent.run', runQuickTask),
        vscode.commands.registerCommand('deepseek-agent.commit', smartCommit),
        vscode.commands.registerCommand('deepseek-agent.review', codeReview),
        vscode.commands.registerCommand('deepseek-agent.fix', fixSelected),
        vscode.commands.registerCommand('deepseek-agent.docs', lookupDocs),
        vscode.commands.registerCommand('deepseek-agent.stop', stopAgent),
    );

    // 监听文档变化（自动刷新诊断）
    ctx.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(e => {
            // 可选：自动触发 lint 检查
        })
    );

    console.log('[DeepSeek Agent] 扩展已激活');
}

/**
 * 打开对话面板
 */
async function openChatPanel() {
    const apiKey = getApiKey();
    if (!apiKey) {
        const selected = await vscode.window.showWarningMessage(
            'DeepSeek API Key 未设置。请在设置中配置或在终端设置 DEEPSEEK_API_KEY 环境变量。',
            '打开设置',
            '取消'
        );
        if (selected === '打开设置') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'deepseek.agent');
        }
        return;
    }

    const panel = vscode.window.createWebviewPanel(
        'deepseek-agent',
        'DeepSeek Agent',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [context.extensionPath],
        }
    );

    const html = getWebviewHtml(panel.webview);
    panel.webview.html = html;

    // 处理来自 webview 的消息
    panel.webview.onDidReceiveMessage(async message => {
        const { type, payload } = message;

        if (type === 'run') {
            await executeAgentTask(panel, payload.task, payload.mode);
        }
    });

    // 输出激活提示
    vscode.window.showInformationMessage('DeepSeek Agent 已就绪。输入任务开始。');
}

/**
 * 执行 Agent 任务（流式输出到终端）
 */
async function executeAgentTask(panel, task, mode) {
    const apiKey = getApiKey();
    const projectPath = getConfig().get('projectPath') || vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || '.';

    // 创建输出通道
    const channel = vscode.window.createOutputChannel(`DeepSeek Agent`);
    channel.show(true);
    channel.appendLine(`[DeepSeek Agent] 任务: ${task}\n`);

    try {
        const { spawn } = require('child_process');
        const pythonExe = process.platform === 'win32'
            ? (process.env.PYTHON_PATH || 'python')
            : 'python3';

        // 尝试直接调用 Agent 模块（stdio 流式）
        const child = spawn(pythonExe, [
            '-m', 'deepseek_agent',
            'run',
            task,
            '--project', projectPath,
            '--model', getConfig().get('model') || 'deepseek-chat',
            '--mode', mode || 'react',
        ], {
            env: { ...process.env, DEEPSEEK_API_KEY: apiKey },
            cwd: projectPath,
        });

        child.stdout.on('data', data => {
            const text = data.toString();
            channel.append(text);
            panel.webview.postMessage({ type: 'stream', text });
        });

        child.stderr.on('data', data => {
            channel.appendLine(`[stderr] ${data.toString()}`);
        });

        child.on('close', code => {
            channel.appendLine(`\n[完成] 退出码: ${code}`);
            panel.webview.postMessage({ type: 'done', exitCode: code });
        });

    } catch (err) {
        channel.appendLine(`[错误] ${err.message}`);
        panel.webview.postMessage({ type: 'error', message: err.message });
    }
}

/**
 * 快速任务（输入框）
 */
async function runQuickTask() {
    const task = await vscode.window.showInputBox({
        prompt: '输入 DeepSeek Agent 任务',
        placeHolder: '例如：修复当前文件的 lint 错误',
    });
    if (!task) return;

    const mode = await vscode.window.showQuickPick(['react', 'plan'], {
        placeHolder: '选择 Agent 模式',
    });
    if (!mode) return;

    const panel = vscode.window.createWebviewPanel(
        'deepseek-agent', 'DeepSeek Agent', vscode.ViewColumn.Beside,
        { enableScripts: true, retainContextWhenHidden: true }
    );
    const html = getWebviewHtml(panel.webview);
    panel.webview.html = html;

    await executeAgentTask(panel, task, mode);
}

/**
 * 智能提交
 */
async function smartCommit() {
    const channel = vscode.window.createOutputChannel('DeepSeek Commit');
    channel.show(true);
    channel.appendLine('[DeepSeek Agent] 智能提交分析中...');

    // 获取当前 Git 状态
    const git = vscode.extensions.getExtension('vscode.git')?.exports?.getAPI(1);
    if (!git) {
        vscode.window.showWarningMessage('Git 扩展未激活');
        return;
    }

    const repo = await git.openRepository(vscode.workspace.workspaceFolders?.[0]?.uri);
    if (!repo) return;

    const status = await repo.status();
    const changed = status.changed
        .concat(status.notifies)
        .concat(status.renamed)
        .map(f => f.uri.fsPath);

    if (changed.length === 0) {
        vscode.window.showInformationMessage('没有变更需要提交');
        return;
    }

    // 读取变更内容
    const diffs = await Promise.all(
        changed.slice(0, 10).map(f => getFileDiff(repo, f))
    );

    channel.appendLine(`变更文件: ${changed.join(', ')}`);
    channel.appendLine('\n请在 DeepSeek Agent 对话中查看分析结果。');
}

/**
 * 代码审查
 */
async function codeReview() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc = editor.document;
    const selection = editor.selection;
    const text = selection.isEmpty ? doc.getText() : doc.getText(selection);

    if (!text.trim()) {
        vscode.window.showInformationMessage('请先选中要审查的代码');
        return;
    }

    const task = `审查以下代码，指出潜在问题和改进建议：\n\n\`\`\`\n${text}\n\`\`\``;
    await runQuickTaskWithContext(task);
}

/**
 * 修复选中代码
 */
async function fixSelected() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc = editor.document;
    const selection = editor.selection;
    const text = selection.isEmpty ? '' : doc.getText(selection);

    const task = text
        ? `修复以下代码中的问题：\n\n\`\`\`\n${text}\n\`\`\``
        : `修复当前文件 ${doc.fileName} 中的问题`;

    await runQuickTaskWithContext(task);
}

/**
 * 查文档
 */
async function lookupDocs() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc = editor.document;
    // 尝试获取当前单词
    const word = doc.getText(editor.selection.isEmpty
        ? doc.getWordRangeAtPosition(editor.selection.active)
        : editor.selection
    );

    if (!word) {
        vscode.window.showInformationMessage('请将光标放在要查询的代码上');
        return;
    }

    const task = `查询 ${word} 的官方文档，返回关键 API 用法和示例`;
    await runQuickTaskWithContext(task);
}

/**
 * 停止 Agent
 */
async function stopAgent() {
    vscode.window.showInformationMessage('Agent 已停止');
}

/**
 * 带上下文的快速任务
 */
async function runQuickTaskWithContext(task) {
    const mode = 'react';
    const panel = vscode.window.createWebviewPanel(
        'deepseek-agent', 'DeepSeek Agent', vscode.ViewColumn.Beside,
        { enableScripts: true }
    );
    panel.webview.html = getWebviewHtml(panel.webview);
    await executeAgentTask(panel, task, mode);
}

/**
 * 获取文件 Diff
 */
async function getFileDiff(repo, filePath) {
    try {
        const uri = vscode.Uri.file(filePath);
        const diff = await repo.diffIndexEntry(uri);
        return diff;
    } catch {
        return '';
    }
}

/**
 * Webview HTML
 */
function getWebviewHtml(webview) {
    const nonce = Date.now().toString(36);
    return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'nonce-${nonce}'; style-src 'self' 'unsafe-inline';">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1e1e1e; color: #d4d4d4; height: 100vh; display: flex; flex-direction: column; }
.header { background: #252526; border-bottom: 1px solid #3c3c3c; padding: 12px 16px; display: flex; align-items: center; gap: 8px; }
.header h1 { font-size: 14px; color: #ccc; font-weight: normal; }
.status { font-size: 12px; color: #888; margin-left: auto; }
.messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.msg { padding: 10px 14px; border-radius: 6px; max-width: 90%; line-height: 1.5; font-size: 13px; white-space: pre-wrap; }
.msg.user { background: #094771; align-self: flex-end; }
.msg.agent { background: #2d2d2d; border: 1px solid #3c3c3c; align-self: flex-start; }
.msg.tool { background: #1a3a1a; border-left: 3px solid #4ec9b0; font-family: monospace; font-size: 12px; }
.msg.error { background: #3d1a1a; border-left: 3px solid #f14c4c; }
.input-area { background: #252526; border-top: 1px solid #3c3c3c; padding: 12px 16px; display: flex; gap: 8px; }
.input-area input { flex: 1; background: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 8px 12px; color: #d4d4d4; font-size: 13px; outline: none; }
.input-area button { background: #0e639c; border: none; border-radius: 4px; padding: 8px 16px; color: white; font-size: 13px; cursor: pointer; }
.input-area button:hover { background: #1177bb; }
</style>
</head>
<body>
<div class="header">
  <h1>🧠 DeepSeek Code Agent</h1>
  <span class="status" id="status">就绪</span>
</div>
<div class="messages" id="messages"></div>
<div class="input-area">
  <input id="input" placeholder="输入任务，按 Enter 发送..." autofocus />
  <button onclick="send()">发送</button>
</div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
const messages = document.getElementById('messages');
const input = document.getElementById('input');
const status = document.getElementById('status');

function append(type, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + type;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function send() {
  const text = input.value.trim();
  if (!text) return;
  append('user', text);
  input.value = '';
  status.textContent = '思考中...';
  vscode.postMessage({ type: 'run', payload: { task: text, mode: 'react' } });
}

input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

window.addEventListener('message', event => {
  const { type, text, message, exitCode } = event.data;
  if (type === 'stream') {
    append('agent', text);
    status.textContent = '处理中...';
  } else if (type === 'done') {
    status.textContent = exitCode === 0 ? '✅ 完成' : '⚠️ 已结束';
  } else if (type === 'error') {
    append('error', '错误: ' + message);
  }
});
</script>
</body>
</html>`;
}

function deactivate() {}

module.exports = { activate, deactivate };
