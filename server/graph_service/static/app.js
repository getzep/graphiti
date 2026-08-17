const state = {
  status: null,
  wikis: [],
  currentWikiId: localStorage.getItem('vaka.currentWikiId'),
  sources: [],
  jobs: [],
  connections: [],
  connectionSelection: { feishu: null, meego: null },
  oauthPopups: new Map(),
  resourcePickers: {
    feishu: {
      connectionId: null, parentId: '', path: [], items: [], nextPage: null,
      root: false, folder: null, documents: new Map(),
    },
    meego: {
      connectionId: null,
      projects: [], project: null, projectsLoaded: false, projectsLoading: false, projectsError: null,
      views: [], view: null, viewQuery: '', viewsLoading: false, viewsError: null, viewRequest: 0,
    },
  },
  sourceFilter: 'all',
  sourceQuery: '',
  uploadSource: null,
  uploadFiles: [],
  graph: { nodes: [], edges: [], selected: null, loadedFor: null },
  wikiMode: 'directory',
  entityTab: 'overview',
  wikiTask: null,
  wikiPlan: null,
  createFlow: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function api(path, options = {}) {
  const { headers = {}, ...requestOptions } = options;
  const response = await fetch(path, {
    ...requestOptions,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
    } catch (_) {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function toast(message, type = 'info') {
  const element = document.createElement('div');
  element.className = `toast ${type}`;
  element.textContent = message;
  $('#toast-region').append(element);
  setTimeout(() => element.remove(), 4200);
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(date);
}

function kindName(kind) {
  return { local: '本地文件', feishu: '飞书', meego: 'MeeGo' }[kind] || kind;
}

function connectionProvider(connection) {
  const provider = String(connection.provider || connection.kind || '').toLowerCase();
  return provider === 'lark' ? 'feishu' : provider;
}

function connectionLabel(connection) {
  const metadata = connection.metadata || {};
  return connection.display_name || connection.account_name || connection.name
    || connection.account || metadata.display_name || metadata.account_name || metadata.email
    || `连接 ${String(connection.id || '').slice(0, 8)}`;
}

function connectionsFor(provider) {
  const expectedHost = provider === 'meego' ? state.status?.oauth?.hosts?.meego : null;
  return state.connections.filter((connection) => (
    connectionProvider(connection) === provider
    && (!expectedHost || connection.tenant_id === expectedHost)
  ));
}

function currentWiki() {
  return state.wikis.find((wiki) => wiki.id === state.currentWikiId) || null;
}

function renderWikiControls() {
  const select = $('#wiki-select');
  if (!state.wikis.length) {
    select.innerHTML = '<option value="">请先创建 Wiki</option>';
    select.disabled = true;
  } else {
    select.disabled = false;
    select.innerHTML = state.wikis.map((wiki) => (
      `<option value="${escapeHTML(wiki.id)}">${escapeHTML(wiki.name)} · ${escapeHTML(wiki.candidate_status)}</option>`
    )).join('');
    select.value = state.currentWikiId;
  }
  const wiki = currentWiki();
  // Adding data is also the primary first-run entry. Keep it actionable even
  // before a Wiki exists and continue the upload flow after Wiki creation.
  $('#add-source-button').disabled = false;
  $('#task-link').disabled = !wiki;
  $('#plan-link').disabled = !wiki;
  $('#mcp-link').disabled = !wiki;
  if (wiki) {
    const labels = { empty: '尚未构建', building: '构建中', ready: '等待发布', failed: '构建失败' };
    const statusLabel = wiki.published_group_id === wiki.candidate_group_id
      ? '已完成'
      : labels[wiki.candidate_status] || '已完成';
    $('#task-link-label').textContent = `${statusLabel} · 查看任务`;
    $('#plan-link-label').textContent = `策略 v${wiki.plan_version || 1}`;
  } else {
    $('#task-link-label').textContent = '尚未构建 · 查看任务';
    $('#plan-link-label').textContent = '策略 v1';
  }
}

function selectedConnection(provider) {
  const id = state.connectionSelection[provider];
  return connectionsFor(provider).find((connection) => connection.id === id) || null;
}

const pageLabels = { library: '文件库', wiki: '我的 Wiki' };

function navigate(view) {
  $$('.view').forEach((element) => element.classList.toggle('active', element.id === `view-${view}`));
  $$('.center-tabs [data-nav]').forEach((element) => element.classList.toggle('active', element.dataset.nav === view));
  location.hash = view;
  $('.app-rail').classList.remove('open');
  if (view === 'wiki' && state.wikiMode === 'graph') setTimeout(resizeGraphCanvas, 20);
}

document.addEventListener('click', (event) => {
  const nav = event.target.closest('[data-nav]');
  if (nav) {
    event.preventDefault();
    navigate(nav.dataset.nav);
  }
});

$('#mobile-menu').addEventListener('click', () => $('.app-rail').classList.toggle('open'));
$$('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));

function renderStatus() {
  if (!state.status) return;
  const { database, llm } = state.status;
  const ready = database.ready !== false && llm.configured;
  $('#system-chips').classList.toggle('error', !ready);
  $('#system-chips').innerHTML = `<i></i><span>${ready ? '知识底座已就绪' : '知识底座配置不完整'}</span>`;
  const wiki = currentWiki();
  const mcpUrl = wiki?.mcp_url || '尚未创建 Wiki';
  $('#mcp-url').textContent = mcpUrl;
  $('#mcp-config').textContent = JSON.stringify({
    mcpServers: { vakaWiki: { url: mcpUrl } },
  }, null, 2);
  renderWikiControls();
  renderWiki();
}

function sourceDetail(source) {
  if (source.kind === 'local') return '浏览器上传 · 内容指纹去重 · 文件留在服务端数据目录';
  const connection = state.connections.find((item) => item.id === source.connection_id);
  const account = connection ? `${connectionLabel(connection)} · ` : '';
  if (source.kind === 'feishu') {
    if (source.config.root_folder) return `${account}整个我的空间`;
    if (source.config.folder_token) return `${account}已选择文件夹`;
    return `${account}${(source.config.document_tokens || []).length} 个文档`;
  }
  if (source.config.view_id) {
    const project = source.config.project_name || source.config.project_key;
    const view = source.config.view_name || source.config.view_id;
    return `${account}${project} · ${view} · 需求视图`;
  }
  return `${account}项目 ${source.config.project_key || '—'} · 自动同步全部可读工作项`;
}

function renderSources() {
  const query = state.sourceQuery.toLocaleLowerCase();
  const filtered = state.sources.filter((source) => (
    (state.sourceFilter === 'all' || source.kind === state.sourceFilter)
    && (!query || source.name.toLocaleLowerCase().includes(query))
  ));
  const grid = $('#source-grid');
  if (!filtered.length) {
    grid.innerHTML = currentWiki()
      ? state.sources.length
        ? '<div class="table-empty">没有符合当前搜索或筛选条件的数据。</div>'
        : '<div class="table-empty">还没有数据。点击右上角“添加数据”，可上传本地文件或连接飞书、MeeGo。</div>'
      : '<div class="table-empty"><strong>还没有 Wiki，数据暂时无处归属</strong><br>先创建一个 Wiki，完成后会自动打开添加数据窗口。<br><button class="primary-button" data-action="start-create-upload">创建 Wiki 并添加数据</button></div>';
    return;
  }
  grid.innerHTML = filtered.map((source) => `
    <article class="source-card" data-source-id="${source.id}">
      <span class="source-icon">${source.kind === 'local' ? 'FILE' : source.kind === 'feishu' ? 'LARK' : 'MEE'}</span>
      <div class="source-title"><h3>${escapeHTML(source.name)}</h3><small>${escapeHTML(sourceDetail(source))}</small></div>
      <span class="source-kind">${escapeHTML(kindName(source.kind))}</span>
      <span class="source-detail">${escapeHTML(source.last_error || '可被当前 Wiki 使用')}</span>
      <span class="source-time">${formatDate(source.last_sync_at || source.created_at)}</span>
      <div class="source-actions">
        <span class="source-status ${source.status === 'error' ? 'error' : ''}">${source.status === 'syncing' ? '同步中' : source.status === 'error' ? '异常' : source.enabled ? '可用' : '已停用'}</span>
        ${source.kind === 'local' ? '<button data-source-action="upload">上传</button>' : ''}
        <button data-source-action="toggle">${source.enabled ? '停用' : '启用'}</button>
        <button class="danger" data-source-action="delete">移除</button>
      </div>
    </article>
  `).join('');
}

function renderJobs() {
  const names = Object.fromEntries(state.sources.map((source) => [source.id, source.name]));
  const visibleJobs = state.jobs.filter((job) => job.source_id in names);
  const recent = visibleJobs.slice(0, 5);
  $('#recent-jobs').innerHTML = recent.length ? recent.map((job) => (
    `<p>${escapeHTML(names[job.source_id] || 'Wiki 构建')} · ${escapeHTML(job.status)}</p>`
  )).join('') : '暂无任务';
}

const typeNames = {
  Project: '项目', Product: '产品', ProductModule: '产品模块', ProductFeature: '产品功能',
  Version: '版本', Requirement: '产品需求', Defect: '缺陷', Person: '人员', Entity: '其他实体',
};

function nodeType(node) {
  return (node.labels || []).find((label) => label !== 'Entity') || 'Entity';
}

function relationRows(node) {
  if (!node) return [];
  const byId = Object.fromEntries(state.graph.nodes.map((item) => [item.id, item]));
  return state.graph.edges.filter((edge) => edge.source === node.id || edge.target === node.id)
    .map((edge) => {
      const outgoing = edge.source === node.id;
      const other = byId[outgoing ? edge.target : edge.source];
      return { ...edge, outgoing, other };
    });
}

function renderWikiTree() {
  const container = $('#wiki-tree');
  const query = $('#tree-search').value.trim().toLocaleLowerCase();
  const groups = new Map();
  state.graph.nodes.forEach((node) => {
    if (query && !node.name.toLocaleLowerCase().includes(query)) return;
    const type = nodeType(node);
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type).push(node);
  });
  if (!groups.size) {
    container.innerHTML = `<div class="sidebar-empty">${currentWiki()?.published_group_id ? '没有匹配的实体' : '构建完成后将在这里显示实体目录'}</div>`;
    return;
  }
  container.innerHTML = [...groups.entries()].sort(([a], [b]) => (typeNames[a] || a).localeCompare(typeNames[b] || b)).map(([type, nodes]) => `
    <div class="tree-group">
      <button class="tree-group-title"><span>▱ ${escapeHTML(typeNames[type] || type)}</span><small>${nodes.length}</small></button>
      <div class="tree-items">${nodes.sort((a, b) => a.name.localeCompare(b.name)).map((node) => `
        <button class="tree-item ${state.graph.selected?.id === node.id ? 'active' : ''}" data-node-id="${escapeHTML(node.id)}">${escapeHTML(node.name)}</button>
      `).join('')}</div>
    </div>
  `).join('');
}

function renderEntityBody() {
  const node = state.graph.selected;
  const body = $('#entity-body');
  if (!node) {
    body.innerHTML = '<div class="empty-note">从左侧目录选择一个实体</div>';
    return;
  }
  const relations = relationRows(node);
  const attributes = Object.entries(node.attributes || {}).filter(([, value]) => value !== null && value !== '' && (!Array.isArray(value) || value.length));
  if (state.entityTab === 'attributes') {
    body.innerHTML = attributes.length ? `<dl class="attribute-list">${attributes.map(([key, value]) => `<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(Array.isArray(value) ? value.join('、') : value)}</dd>`).join('')}</dl>` : '<div class="empty-note">来源中没有可确认的结构化属性</div>';
  } else if (state.entityTab === 'relations') {
    body.innerHTML = relations.length ? `<div class="relation-list">${relations.map((edge) => `<article class="relation-card"><strong>${escapeHTML(edge.outgoing ? `${node.name} → ${edge.other?.name || '未知实体'}` : `${edge.other?.name || '未知实体'} → ${node.name}`)}</strong><p>${escapeHTML(edge.fact || edge.name)}</p></article>`).join('')}</div>` : '<div class="empty-note">当前实体没有已发布关系</div>';
  } else if (state.entityTab === 'sources') {
    body.innerHTML = `<section class="entity-section"><h3>来源与权限</h3><p>当前内容来自 ${state.sources.length} 个已关联数据源。MVP 按 Wiki 实例读取已发布版本；来源级证据定位将在 Graphiti Crosswalk 之上继续补齐。</p></section>${state.sources.map((source) => `<article class="relation-card"><strong>${escapeHTML(source.name)}</strong><p>${escapeHTML(kindName(source.kind))} · 最近同步 ${formatDate(source.last_sync_at)}</p></article>`).join('')}`;
  } else if (state.entityTab === 'changes') {
    const wiki = currentWiki();
    body.innerHTML = `<section class="entity-section"><h3>变更记录</h3><p>当前读取发布版本：${formatDate(wiki?.published_at)}。自动构建使用独立 Candidate，失败不会覆盖该版本。</p></section><div class="empty-note">还没有人工更正记录</div>`;
  } else {
    const highlights = attributes.slice(0, 5);
    body.innerHTML = `
      <section class="entity-section"><h3>概览</h3><p>${escapeHTML(node.summary || '当前来源没有生成可确认的实体摘要。')}</p></section>
      <section class="entity-section"><h3>关键属性</h3>${highlights.length ? `<dl class="attribute-list">${highlights.map(([key, value]) => `<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(Array.isArray(value) ? value.join('、') : value)}</dd>`).join('')}</dl>` : '<p>暂无结构化属性。</p>'}</section>
      <section class="entity-section"><h3>关联实体</h3>${relations.length ? `<div class="relation-list">${relations.slice(0, 10).map((edge) => `<article class="relation-card"><strong>${escapeHTML(edge.other?.name || '未知实体')}</strong><p>${escapeHTML(edge.fact || edge.name)}</p></article>`).join('')}</div>` : '<p>暂无已发布关系。</p>'}</section>`;
  }
}

function renderEntityPage() {
  const node = state.graph.selected;
  const wiki = currentWiki();
  $('#breadcrumbs').textContent = node ? `${wiki?.name || 'Wiki'} › ${typeNames[nodeType(node)] || nodeType(node)}` : wiki?.name || 'Wiki';
  $('#entity-title').textContent = node?.name || wiki?.name || 'Wiki';
  $('#entity-subtitle').textContent = node
    ? `${typeNames[nodeType(node)] || nodeType(node)} · 更新于 ${formatDate(node.created_at)}`
    : `${state.graph.nodes.length} 个实体 · ${state.graph.edges.length} 条关系`;
  renderEntityBody();
}

function renderWiki() {
  const wiki = currentWiki();
  renderWikiTree();
  const placeholder = $('#wiki-placeholder');
  const buildState = $('#build-state');
  const entityPage = $('#entity-page');
  const graphPage = $('#graph-page');
  [placeholder, buildState, entityPage, graphPage].forEach((element) => element.classList.add('hidden'));
  if (!wiki) {
    const action = placeholder.querySelector('[data-action]');
    action.dataset.action = 'start-create-upload';
    action.textContent = '创建 Wiki 并添加数据';
    placeholder.querySelector('h2').textContent = '用对话创建第一个 Wiki';
    placeholder.querySelector('p').textContent = '告诉 Vaka Copilot 你想管理什么知识。它会确认目标和数据范围，生成项目 Wiki 计划。';
    placeholder.classList.remove('hidden');
    return;
  }
  if (state.wikiMode === 'graph' && wiki.published_group_id) {
    graphPage.classList.remove('hidden');
    setTimeout(resizeGraphCanvas, 20);
    return;
  }
  if (!wiki.published_group_id) {
    if (wiki.candidate_status === 'building' || wiki.candidate_status === 'failed') {
      buildState.classList.remove('hidden');
      const failed = wiki.candidate_status === 'failed';
      $('#build-state-title').textContent = failed ? 'Wiki 构建未完成' : 'Wiki 内容构建中';
      $('#build-state-copy').textContent = failed
        ? '候选结果没有发布。请查看任务错误，修复数据或授权后重新开始。'
        : '任务会在后台继续运行。完成并通过检查后，将自动发布实体目录与图谱。';
    } else {
      placeholder.classList.remove('hidden');
      const action = placeholder.querySelector('[data-action]');
      if (state.sources.length) {
        placeholder.querySelector('h2').textContent = `${state.sources.length} 项数据已添加`;
        placeholder.querySelector('p').textContent = '数据已经进入当前 Wiki 的文件库，但构建尚未开始。可以直接恢复构建，无需重新添加。';
        action.textContent = '开始构建';
        action.dataset.action = 'start-build';
      } else {
        placeholder.querySelector('h2').textContent = `为“${wiki.name}”添加数据`;
        placeholder.querySelector('p').textContent = '计划已经生成。添加本地文件、飞书或 MeeGo 数据后，系统会开始异步构建。';
        action.textContent = '添加数据';
        action.dataset.action = 'add-data';
      }
    }
    return;
  }
  if (!state.graph.selected && state.graph.nodes.length) state.graph.selected = state.graph.nodes[0];
  entityPage.classList.remove('hidden');
  renderEntityPage();
}

function resetResourcePicker(provider) {
  const connectionId = state.connectionSelection[provider] || null;
  if (provider === 'feishu') {
    state.resourcePickers.feishu = {
      connectionId, parentId: '', path: [], items: [], nextPage: null,
      loaded: false, loading: false, error: null,
      root: false, folder: null, documents: new Map(),
    };
  } else {
    state.resourcePickers.meego = {
      connectionId,
      projects: [], project: null, projectsLoaded: false, projectsLoading: false,
      projectsError: null, views: [], view: null, viewQuery: '', viewsLoading: false,
      viewsError: null, viewRequest: 0,
    };
  }
  renderResourcePicker(provider);
}

function connectionStatus(connection) {
  const value = String(connection.status || '').toLowerCase();
  if (['expired', 'error', 'revoked', 'disabled'].includes(value)) return ' · 需要重新连接';
  return '';
}

function renderConnectionPanels() {
  ['feishu', 'meego'].forEach((provider) => {
    const providerAvailable = state.status?.oauth?.providers?.[provider] !== false;
    const connectButton = $(`[data-oauth-provider="${provider}"]`);
    const connections = connectionsFor(provider);
    let selectedId = state.connectionSelection[provider];
    if (!connections.some((connection) => connection.id === selectedId)) {
      selectedId = connections[0]?.id || null;
      state.connectionSelection[provider] = selectedId;
    }
    const connection = selectedConnection(provider);
    if (connectButton) {
      connectButton.disabled = provider !== 'feishu' && !providerAvailable;
      connectButton.title = '';
      connectButton.textContent = provider === 'meego' && connection ? '重新连接' : `连接 ${kindName(provider)}`;
    }
    const select = $(`#${provider}-connection`);
    if (select) {
      select.disabled = connections.length === 0;
      select.innerHTML = connections.length
        ? connections.map((item) => `<option value="${escapeHTML(item.id)}">${escapeHTML(connectionLabel(item) + connectionStatus(item))}</option>`).join('')
        : '<option value="">尚未连接账号</option>';
      select.value = selectedId || '';
    }
    if (provider === 'meego') {
      const ready = Boolean(connection) && !connectionStatus(connection);
      connectButton?.classList.toggle('hidden', ready);
      $('#meego-connection-label').textContent = ready ? 'MeeGo 已连接' : 'MeeGo 尚未连接';
      const rawName = connection ? connectionLabel(connection) : '';
      const accountName = rawName.startsWith('MeeGo 账号 ·') ? '当前账号' : rawName;
      $('#meego-account-detail').textContent = connection
        ? `${accountName} · ${connection.tenant_id || state.status?.oauth?.hosts?.meego || 'MeeGo'}`
        : '连接后同步你可见的需求视图';
      const availability = $('#meego-connection-state');
      availability.textContent = ready ? '可用' : '未连接';
      availability.classList.toggle('ready', ready);
    } else {
      $('#feishu-connection-label').textContent = connection
        ? connectionLabel(connection) + connectionStatus(connection)
        : '尚未连接';
    }
    const browse = $(`[data-resource-load="${provider}"]`);
    if (browse) browse.disabled = !connection;
    const picker = state.resourcePickers[provider];
    if (picker.connectionId !== selectedId) resetResourcePicker(provider);
    const currentPicker = state.resourcePickers[provider];
    if (provider === 'meego' && sourceForm.elements.kind?.value === 'meego' && connection
      && !currentPicker.projectsLoaded && !currentPicker.projectsLoading) {
      loadMeeGoProjects();
    }
  });
}

async function loadConnections({ quiet = false, preferredProvider = null, preferredConnectionId = null } = {}) {
  try {
    const payload = await api('/api/connections');
    const connections = Array.isArray(payload) ? payload : (payload.items || payload.connections || []);
    state.connections = connections.map((connection) => ({
      ...connection,
      id: String(connection.id),
      metadata: connection.metadata || {},
    }));
    if (preferredProvider && preferredConnectionId
      && state.connections.some((connection) => connection.id === preferredConnectionId
        && connectionProvider(connection) === preferredProvider)) {
      state.connectionSelection[preferredProvider] = preferredConnectionId;
    }
    renderConnectionPanels();
    renderSources();
  } catch (error) {
    if (!quiet) toast(`无法读取连接账号：${error.message}`, 'error');
  }
}

function resourceTypeName(type) {
  return {
    folder: '文件夹', doc: '文档', docx: '文档', sheet: '表格', bitable: '多维表格',
    file: '文件', project: '项目', work_item_type: '工作项类型', issue_type: '工作项类型',
  }[type] || type || '资源';
}

function itemKey(item, kind) {
  const metadata = item.metadata || {};
  if (kind === 'project') return String(metadata.project_key || metadata.key || item.id);
  if (kind === 'work_item_type') {
    return String(metadata.work_item_type_key || metadata.type_key || metadata.key || item.id);
  }
  return String(item.id);
}

function isFeishuFolder(item) {
  return item.type === 'folder';
}

function renderSelection(provider) {
  const picker = state.resourcePickers[provider];
  const container = $(`#${provider}-selection`);
  if (provider === 'feishu') {
    let chips = '';
    if (picker.root) chips = '<span class="selection-chip">整个我的空间</span>';
    else if (picker.folder) chips = `<span class="selection-chip">文件夹 · ${escapeHTML(picker.folder.name)}</span>`;
    else if (picker.documents.size) {
      chips = [...picker.documents.values()].map((item) => `<span class="selection-chip">${escapeHTML(item.name)}</span>`).join('');
    }
    container.innerHTML = chips
      ? `<div class="selection-chips">${chips}</div><button type="button" data-clear-selection="feishu">清除选择</button>`
      : '尚未选择资源';
  }
}

function resourceRow(provider, item, index) {
  const picker = state.resourcePickers[provider];
  const folder = provider === 'feishu' && isFeishuFolder(item);
  let selectAction = '';
  let selected = false;
  if (provider === 'feishu' && folder && item.selectable) {
    selected = picker.folder?.id === item.id;
    selectAction = `<button type="button" data-select-folder="${index}">${selected ? '已选择' : '选择目录'}</button>`;
  } else if (provider === 'feishu' && !folder && item.selectable) {
    selected = picker.documents.has(item.id);
    selectAction = `<button type="button" data-toggle-document="${index}">${selected ? '取消' : '选择'}</button>`;
  }
  const openAction = item.has_children
    ? `<button type="button" data-resource-open="${index}">打开</button>` : '';
  return `<div class="resource-row ${selected ? 'selected' : ''}">
    <span class="resource-kind">${escapeHTML(resourceTypeName(item.type))}</span>
    <div><strong>${escapeHTML(item.name)}</strong><small>${item.has_children ? '包含下级资源' : escapeHTML(item.metadata?.description || '')}</small></div>
    <div class="resource-row-actions">${selectAction}${openAction}</div>
  </div>`;
}

function renderMeeGoSelectors() {
  const picker = state.resourcePickers.meego;
  const projectSelect = $('#meego-project-select');
  projectSelect.disabled = !state.connectionSelection.meego || picker.projectsLoading
    || Boolean(picker.projectsError) || !picker.projects.length;
  if (!state.connectionSelection.meego) {
    projectSelect.innerHTML = '<option value="">请先连接 MeeGo</option>';
  } else if (picker.projectsLoading && !picker.projects.length) {
    projectSelect.innerHTML = '<option value="">正在读取可见项目…</option>';
  } else if (picker.projectsError) {
    projectSelect.innerHTML = `<option value="">${escapeHTML(picker.projectsError)}</option>`;
  } else {
    projectSelect.innerHTML = '<option value="">请选择产品 / 项目</option>' + picker.projects.map((item) => (
      `<option value="${escapeHTML(item.id)}">${escapeHTML(item.name)} (${escapeHTML(item.key)})</option>`
    )).join('');
    projectSelect.value = picker.project?.id || '';
  }

  const queryInput = $('#meego-view-query');
  queryInput.disabled = !picker.project;
  if (queryInput.value !== picker.viewQuery) queryInput.value = picker.viewQuery;
  const viewSelect = $('#meego-view-select');
  viewSelect.disabled = !picker.project || picker.viewsLoading || Boolean(picker.viewsError)
    || !picker.views.length;
  if (!picker.project) {
    viewSelect.innerHTML = '<option value="">请先选择产品 / 项目</option>';
  } else if (picker.viewsLoading) {
    viewSelect.innerHTML = '<option value="">正在读取需求视图…</option>';
  } else if (picker.viewsError) {
    viewSelect.innerHTML = `<option value="">${escapeHTML(picker.viewsError)}</option>`;
  } else if (!picker.views.length) {
    viewSelect.innerHTML = '<option value="">没有匹配的需求视图，请输入名称查找</option>';
  } else {
    viewSelect.innerHTML = '<option value="">请选择导入范围</option>' + picker.views.map((item) => (
      `<option value="${escapeHTML(item.id)}">${escapeHTML(item.name)}</option>`
    )).join('');
    viewSelect.value = picker.view?.id || '';
  }

  const title = $('#meego-scope-title');
  const detail = $('#meego-scope-detail');
  if (picker.projectsError || picker.viewsError) {
    title.textContent = '暂时无法读取 MeeGo 导入范围';
    detail.textContent = picker.projectsError || picker.viewsError;
  } else if (picker.view) {
    title.textContent = `${picker.project.name} · ${picker.view.name}`;
    detail.textContent = '将精确同步该需求视图当前筛选出的工作项，不会导入整个项目。';
  } else if (picker.project) {
    title.textContent = `${picker.project.name} · 请选择需求视图`;
    detail.textContent = '下拉框展示默认候选；如果视图较多，可按视图名称查找。';
  } else {
    title.textContent = '选择产品 / 项目及需求视图';
    detail.textContent = '只同步选中视图当前筛选出的需求；视图筛选发生变化后，下次同步会自动跟随。';
  }
}

function renderResourcePicker(provider) {
  const picker = state.resourcePickers[provider];
  if (provider === 'meego') {
    renderMeeGoSelectors();
    return;
  }
  const browser = $(`#${provider}-resource-browser`);
  renderSelection(provider);
  if (!picker.loaded && !picker.loading && !picker.error) {
    browser.classList.add('hidden');
    browser.innerHTML = '';
    return;
  }
  browser.classList.remove('hidden');
  if (picker.loading && !picker.items.length) {
    browser.innerHTML = '<div class="resource-browser-state">正在读取可见资源…</div>';
    return;
  }
  if (picker.error) {
    browser.innerHTML = `<div class="resource-browser-state error">${escapeHTML(picker.error)}</div>`;
    return;
  }
  const crumbs = ['<button type="button" data-resource-home>根目录</button>']
    .concat(picker.path.map((entry) => `<span>›</span><span>${escapeHTML(entry.name)}</span>`)).join('');
  const scopeAction = provider === 'feishu'
    ? (picker.path.length
      ? '<button type="button" data-select-current-folder>选择当前目录</button>'
      : '<button type="button" data-select-root>选择整个空间</button>')
    : '';
  const rows = picker.items.length
    ? picker.items.map((item, index) => resourceRow(provider, item, index)).join('')
    : '<div class="resource-browser-state">当前层级没有可选资源</div>';
  const more = picker.nextPage !== null && picker.nextPage !== undefined && picker.nextPage !== ''
    ? `<button class="resource-more" type="button" data-resource-next="${escapeHTML(picker.nextPage)}">加载更多</button>` : '';
  browser.innerHTML = `<div class="resource-browser-toolbar">
      <div class="resource-breadcrumb">${crumbs}</div>
      <div>${picker.path.length ? '<button type="button" data-resource-back>返回上级</button>' : ''}${scopeAction}</div>
    </div><div class="resource-list">${rows}</div>${more}`;
}

async function loadMeeGoProjects() {
  const picker = state.resourcePickers.meego;
  const connectionId = state.connectionSelection.meego;
  if (!connectionId) return;
  picker.projectsLoading = true;
  picker.projectsError = null;
  renderResourcePicker('meego');
  try {
    const projects = new Map();
    let page = '1';
    for (let request = 0; request < 10 && page; request += 1) {
      const payload = await api(
        `/api/connections/${encodeURIComponent(connectionId)}/resources?page=${encodeURIComponent(page)}`,
      );
      (payload.items || []).forEach((item) => {
        const id = String(item.id);
        projects.set(id, {
          ...item,
          id,
          name: String(item.name || id),
          key: String(item.metadata?.simple_name || id),
          metadata: item.metadata || {},
        });
      });
      page = String(payload.next_page || '');
    }
    if (state.connectionSelection.meego !== connectionId) return;
    picker.projects = [...projects.values()];
    picker.projectsLoaded = true;
  } catch (error) {
    if (state.connectionSelection.meego !== connectionId
      || state.resourcePickers.meego !== picker) return;
    picker.projectsError = `无法读取产品 / 项目：${error.message}`;
  } finally {
    if (state.connectionSelection.meego !== connectionId
      || state.resourcePickers.meego !== picker) return;
    picker.projectsLoading = false;
    renderResourcePicker('meego');
  }
}

async function loadMeeGoViews() {
  const picker = state.resourcePickers.meego;
  const connectionId = state.connectionSelection.meego;
  const project = picker.project;
  if (!connectionId || !project) return;
  picker.viewsLoading = true;
  picker.viewsError = null;
  picker.view = null;
  const requestId = ++picker.viewRequest;
  renderResourcePicker('meego');
  try {
    const query = new URLSearchParams({ project_key: project.key, query: picker.viewQuery });
    const payload = await api(
      `/api/connections/${encodeURIComponent(connectionId)}/meego/views?${query}`,
    );
    if (requestId !== picker.viewRequest || picker.project?.id !== project.id) return;
    picker.views = (payload.items || []).map((item) => ({
      id: String(item.id), name: String(item.name || item.id),
    }));
  } catch (error) {
    if (requestId === picker.viewRequest) {
      picker.viewsError = `无法读取需求视图：${error.message}`;
      picker.views = [];
    }
  } finally {
    if (requestId === picker.viewRequest) {
      picker.viewsLoading = false;
      renderResourcePicker('meego');
    }
  }
}

async function loadResources(provider, { parentId = '', path = [], page = 1, append = false } = {}) {
  const connectionId = state.connectionSelection[provider];
  if (!connectionId) return toast(`请先连接${kindName(provider)}`, 'error');
  const picker = state.resourcePickers[provider];
  picker.connectionId = connectionId;
  picker.parentId = parentId;
  picker.path = path;
  picker.loading = true;
  picker.error = null;
  if (!append) picker.items = [];
  renderResourcePicker(provider);
  try {
    const query = new URLSearchParams({ parent_id: parentId || '', page: String(page) });
    const payload = await api(`/api/connections/${encodeURIComponent(connectionId)}/resources?${query}`);
    const items = (payload.items || []).map((item) => ({
      ...item,
      id: String(item.id),
      name: String(item.name || item.id),
      type: String(item.type || ''),
      selectable: item.selectable !== false,
      has_children: Boolean(item.has_children),
      metadata: item.metadata || {},
    }));
    if (append) {
      const merged = new Map(picker.items.map((item) => [item.id, item]));
      items.forEach((item) => merged.set(item.id, item));
      picker.items = [...merged.values()];
    } else picker.items = items;
    picker.nextPage = payload.next_page ?? null;
    picker.loaded = true;
  } catch (error) {
    picker.error = error.message;
    toast(`无法读取资源：${error.message}`, 'error');
  } finally {
    picker.loading = false;
    renderResourcePicker(provider);
  }
}

function selectFeishuRoot() {
  const picker = state.resourcePickers.feishu;
  picker.root = true; picker.folder = null; picker.documents.clear();
  renderResourcePicker('feishu');
}

function selectFeishuFolder(item) {
  const picker = state.resourcePickers.feishu;
  picker.root = false; picker.folder = item; picker.documents.clear();
  renderResourcePicker('feishu');
}

function toggleFeishuDocument(item) {
  const picker = state.resourcePickers.feishu;
  picker.root = false; picker.folder = null;
  if (picker.documents.has(item.id)) picker.documents.delete(item.id);
  else picker.documents.set(item.id, item);
  renderResourcePicker('feishu');
}

async function refreshData({ quiet = false } = {}) {
  if (refreshData.inFlight) return refreshData.inFlight;
  refreshData.inFlight = (async () => {
    try {
      const [status, wikis, jobs] = await Promise.all([
        api('/api/status'), api('/api/wikis'), api('/api/jobs?limit=50'),
      ]);
      state.status = status;
      state.wikis = wikis;
      if (!state.wikis.some((wiki) => wiki.id === state.currentWikiId)) {
        state.currentWikiId = state.wikis[0]?.id || null;
      }
      if (state.currentWikiId) localStorage.setItem('vaka.currentWikiId', state.currentWikiId);
      else localStorage.removeItem('vaka.currentWikiId');
      state.sources = state.currentWikiId
        ? await api(`/api/sources?wiki_id=${encodeURIComponent(state.currentWikiId)}`)
        : [];
      state.jobs = jobs;
      const wiki = currentWiki();
      if (wiki) {
        const [task, plan] = await Promise.all([
          api(`/api/wikis/${wiki.id}/task`), api(`/api/wikis/${wiki.id}/plan`),
        ]);
        state.wikiTask = task;
        state.wikiPlan = plan;
        if (wiki.published_group_id && state.graph.loadedFor !== wiki.published_group_id) {
          const graph = await api(`/api/wikis/${wiki.id}/graph`);
          state.graph.nodes = graph.nodes || [];
          state.graph.edges = graph.edges || [];
          state.graph.selected = state.graph.nodes[0] || null;
          state.graph.loadedFor = wiki.published_group_id;
          initializeGraph(state.graph.nodes, state.graph.edges);
        } else if (!wiki.published_group_id) {
          state.graph = { nodes: [], edges: [], selected: null, loadedFor: null };
        }
      } else {
        state.wikiTask = null;
        state.wikiPlan = null;
        state.graph = { nodes: [], edges: [], selected: null, loadedFor: null };
      }
      renderStatus(); renderSources(); renderJobs();
    } catch (error) {
      if (!quiet) toast(`无法读取服务状态：${error.message}`, 'error');
    } finally {
      refreshData.inFlight = null;
    }
  })();
  return refreshData.inFlight;
}

$$('[data-filter]').forEach((button) => button.addEventListener('click', () => {
  state.sourceFilter = button.dataset.filter;
  $$('[data-filter]').forEach((item) => item.classList.toggle('active', item === button));
  renderSources();
}));
$('#source-search').addEventListener('input', (event) => {
  state.sourceQuery = event.target.value.trim();
  renderSources();
});
$('#tree-search').addEventListener('input', renderWikiTree);
$('#refresh-sources').addEventListener('click', () => refreshData());

const sourceDialog = $('#source-dialog');
const sourceForm = $('#source-form');

function setSourceKind(kind) {
  $$('.kind-fields', sourceForm).forEach((fields) => {
    fields.classList.toggle('hidden', fields.dataset.kindFields !== kind);
  });
  const meego = kind === 'meego';
  $('#source-dialog-kicker').textContent = meego ? 'MeeGo' : '文件库';
  $('#source-dialog-title').textContent = meego ? '添加 MeeGo 数据' : '添加数据';
  const description = $('#source-dialog-description');
  description.textContent = meego ? '通过 Vaka 文件库统一管理' : '';
  description.classList.toggle('hidden', !meego);
  $('#source-generic-fields').classList.toggle('hidden', meego);
  sourceForm.elements.name.required = !meego;
  const submitButton = sourceForm.querySelector('button[type="submit"]');
  if (!submitButton.disabled) {
    submitButton.textContent = meego ? '同步到文件库' : '添加到文件库';
  }
  const picker = state.resourcePickers.meego;
  if (meego && state.connectionSelection.meego
    && !picker.projectsLoaded && !picker.projectsLoading) {
    loadMeeGoProjects();
  }
}

async function openSourceDialog() {
  const wiki = currentWiki();
  if (!wiki) {
    navigate('wiki');
    startCreateFlow('', 'add-data');
    return;
  }
  sourceForm.reset();
  state.uploadFiles = [];
  $('#source-file-input').value = '';
  renderUploadFiles();
  setSourceKind('local');
  $('#source-wiki-name').value = wiki.name;
  resetResourcePicker('feishu');
  resetResourcePicker('meego');
  renderConnectionPanels();
  sourceDialog.showModal();
  await loadConnections({ quiet: true });
}

$('#add-source-button').addEventListener('click', openSourceDialog);
$$('input[name="kind"]', sourceForm).forEach((input) => input.addEventListener('change', () => {
  setSourceKind(input.value);
}));

$$('[data-provider-connection]', sourceForm).forEach((select) => select.addEventListener('change', () => {
  const provider = select.dataset.providerConnection;
  state.connectionSelection[provider] = select.value || null;
  resetResourcePicker(provider);
  renderConnectionPanels();
}));

$('#meego-project-select').addEventListener('change', (event) => {
  const picker = state.resourcePickers.meego;
  picker.project = picker.projects.find((item) => item.id === event.target.value) || null;
  picker.viewQuery = '';
  picker.views = [];
  picker.view = null;
  picker.viewsError = null;
  renderResourcePicker('meego');
  if (picker.project) loadMeeGoViews();
});

$('#meego-view-query').addEventListener('input', (event) => {
  const picker = state.resourcePickers.meego;
  picker.viewQuery = event.target.value;
  picker.view = null;
  renderResourcePicker('meego');
  clearTimeout(picker.viewSearchTimer);
  picker.viewSearchTimer = setTimeout(loadMeeGoViews, 350);
});

$('#meego-view-select').addEventListener('change', (event) => {
  const picker = state.resourcePickers.meego;
  picker.view = picker.views.find((item) => item.id === event.target.value) || null;
  renderResourcePicker('meego');
});

function openOAuthPopup(provider, path) {
  const width = 640; const height = 760;
  const left = Math.max(0, window.screenX + (window.outerWidth - width) / 2);
  const top = Math.max(0, window.screenY + (window.outerHeight - height) / 2);
  const popup = window.open(
    path,
    `graphiti-oauth-${provider}`,
    `popup=yes,width=${width},height=${height},left=${Math.round(left)},top=${Math.round(top)}`,
  );
  if (!popup) {
    toast('浏览器阻止了授权窗口，请允许本站打开弹窗后重试', 'error');
    return null;
  }
  state.oauthPopups.set(provider, popup);
  popup.focus();
  return popup;
}

async function completeOAuthConnection(provider, connectionId) {
  await loadConnections({ preferredProvider: provider, preferredConnectionId: connectionId });
  resetResourcePicker(provider);
  toast(`${kindName(provider)}账号已连接`);
  if (provider === 'feishu') await loadResources(provider);
  else await loadMeeGoProjects();
}

function openOAuth(provider) {
  if (!['feishu', 'meego'].includes(provider)) return;
  if (state.status?.oauth?.providers?.[provider] === false) {
    return toast(`${kindName(provider)} OAuth 连接服务尚未就绪`, 'error');
  }
  const current = state.oauthPopups.get(provider);
  if (current && !current.closed) {
    current.focus();
    return;
  }
  openOAuthPopup(
    provider,
    `/api/oauth/${encodeURIComponent(provider)}/start`,
  );
}

window.addEventListener('message', async (event) => {
  if (event.origin !== window.location.origin) return;
  const payload = event.data;
  if (!payload || typeof payload !== 'object'
    || !['graphiti.oauth.complete', 'graphiti.oauth.error'].includes(payload.type)) return;
  if (!['feishu', 'meego'].includes(payload.provider)) return;
  const popup = state.oauthPopups.get(payload.provider);
  if (!popup || event.source !== popup) return;
  state.oauthPopups.delete(payload.provider);
  if (!popup.closed) popup.close();
  if (payload.type === 'graphiti.oauth.error') {
    const message = typeof payload.message === 'string' && payload.message.trim()
      ? payload.message.trim().slice(0, 500)
      : `${kindName(payload.provider)}授权失败`;
    toast(message, 'error');
    return;
  }
  if (typeof payload.connection_id !== 'string' || !payload.connection_id) return;
  await completeOAuthConnection(payload.provider, payload.connection_id);
});

sourceForm.addEventListener('click', (event) => {
  const target = event.target.closest('button');
  if (!target) return;
  const oauthProvider = target.dataset.oauthProvider;
  if (oauthProvider) return openOAuth(oauthProvider);
  const loadProvider = target.dataset.resourceLoad;
  if (loadProvider) return loadResources(loadProvider);
  const browser = target.closest('[data-resource-picker]');
  const provider = browser?.dataset.resourcePicker;
  if (!provider) return;
  const picker = state.resourcePickers[provider];
  if (target.hasAttribute('data-resource-home')) {
    loadResources(provider);
  } else if (target.hasAttribute('data-resource-back')) {
    const path = picker.path.slice(0, -1);
    loadResources(provider, { parentId: path.at(-1)?.id || '', path });
  } else if (target.hasAttribute('data-resource-next')) {
    loadResources(provider, {
      parentId: picker.parentId, path: picker.path,
      page: target.dataset.resourceNext, append: true,
    });
  } else if (target.hasAttribute('data-resource-open')) {
    const item = picker.items[Number(target.dataset.resourceOpen)];
    if (!item) return;
    loadResources(provider, {
      parentId: item.id,
      path: [...picker.path, { id: item.id, name: item.name, type: item.type }],
    });
  } else if (target.hasAttribute('data-select-root')) {
    selectFeishuRoot();
  } else if (target.hasAttribute('data-select-current-folder')) {
    const current = picker.path.at(-1);
    if (current) selectFeishuFolder(current);
  } else if (target.hasAttribute('data-select-folder')) {
    const item = picker.items[Number(target.dataset.selectFolder)];
    if (item) selectFeishuFolder(item);
  } else if (target.hasAttribute('data-toggle-document')) {
    const item = picker.items[Number(target.dataset.toggleDocument)];
    if (item) toggleFeishuDocument(item);
  } else if (target.hasAttribute('data-clear-selection')) {
    picker.root = false; picker.folder = null; picker.documents.clear();
    renderResourcePicker(provider);
  }
});

$('#source-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  // Event.currentTarget is cleared once an async listener yields. Keep the
  // concrete form reference for the post-request reset and button updates.
  const formElement = event.currentTarget;
  const form = new FormData(formElement);
  const kind = form.get('kind');
  if (kind === 'local' && !state.uploadFiles.length) {
    return toast('请先选择要上传的文件', 'error');
  }
  const config = {};
  const payload = {
    kind,
    name: String(form.get('name') || '').trim(),
    wiki_id: state.currentWikiId,
    config,
  };
  if (kind === 'feishu') {
    const picker = state.resourcePickers.feishu;
    const connectionId = state.connectionSelection.feishu;
    if (!connectionId) return toast('请先连接飞书账号', 'error');
    if (!picker.root && !picker.folder && !picker.documents.size) {
      return toast('请选择飞书空间、文件夹或文档', 'error');
    }
    payload.connection_id = connectionId;
    config.folder_token = picker.folder?.id || '';
    config.root_folder = picker.root;
    config.document_tokens = [...picker.documents.keys()];
    config.document_metadata = Object.fromEntries(
      [...picker.documents.entries()].map(([id, item]) => [id, { type: item.type, name: item.name }]),
    );
    config.recursive = form.get('recursive') === 'on';
  } else if (kind === 'meego') {
    const picker = state.resourcePickers.meego;
    const connectionId = state.connectionSelection.meego;
    if (!connectionId) return toast('请先连接 MeeGo 账号', 'error');
    if (!picker.project) return toast('请选择产品 / 项目', 'error');
    if (!picker.view) return toast('请选择要导入的需求视图', 'error');
    payload.connection_id = connectionId;
    payload.name = `${picker.project.name} · ${picker.view.name}`;
    config.project_key = picker.project.key;
    config.project_name = picker.project.name;
    config.work_item_type_key = 'story';
    config.view_id = picker.view.id;
    config.view_name = picker.view.name;
  }
  const submitButton = formElement.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  submitButton.textContent = kind === 'local' ? '正在上传…' : '正在添加…';
  try {
    const source = await api('/api/sources', {
      method: 'POST', body: JSON.stringify(payload),
    });
    const uploadResult = source.kind === 'local'
      ? await uploadFilesToSource(source, state.uploadFiles)
      : null;
    sourceDialog.close(); formElement.reset();
    state.uploadFiles = [];
    renderUploadFiles();
    setSourceKind('local');
    resetResourcePicker('feishu'); resetResourcePicker('meego');
    toast(uploadResult
      ? `“${source.name}”已添加 ${uploadResult.saved.length} 个文件`
      : `已创建“${source.name}”`);
    await refreshData({ quiet: true });
    await startWikiBuild();
  } catch (error) { toast(error.message, 'error'); }
  finally {
    submitButton.disabled = false;
    submitButton.textContent = kind === 'meego' ? '同步到文件库' : '添加到文件库';
  }
});

$('#source-grid').addEventListener('click', async (event) => {
  const actionButton = event.target.closest('[data-source-action]');
  if (!actionButton) return;
  const card = actionButton.closest('[data-source-id]');
  const source = state.sources.find((item) => item.id === card.dataset.sourceId);
  if (!source) return;
  const action = actionButton.dataset.sourceAction;
  if (action === 'upload') return openUpload(source);
  try {
    if (action === 'sync' || action === 'full-sync') {
      const job = await api(`/api/sources/${source.id}/sync`, { method: 'POST', body: JSON.stringify({ full: action === 'full-sync' }) });
      toast(`同步任务 ${job.id.slice(0, 8)} 已进入队列`);
    } else if (action === 'toggle') {
      await api(`/api/sources/${source.id}`, { method: 'PATCH', body: JSON.stringify({ enabled: !source.enabled }) });
      toast(source.enabled ? '数据源已停用' : '数据源已启用');
    } else if (action === 'delete') {
      if (!confirm(`删除“${source.name}”的数据源定义和同步状态？已上传文件与 Neo4j 历史不会删除。`)) return;
      await api(`/api/sources/${source.id}`, { method: 'DELETE' });
      toast('数据源定义已删除');
    }
    await refreshData({ quiet: true });
  } catch (error) { toast(error.message, 'error'); }
});

async function quickLocal() {
  if (!currentWiki()) return startCreateFlow();
  let source = state.sources.find((item) => item.kind === 'local' && item.enabled);
  if (!source) {
    try {
      source = await api('/api/sources', { method: 'POST', body: JSON.stringify({ kind: 'local', name: '本地资料库', wiki_id: state.currentWikiId, config: {} }) });
      await refreshData({ quiet: true });
    } catch (error) { return toast(error.message, 'error'); }
  }
  openUpload(source);
}
$$('[data-action="quick-local"]').forEach((button) => button.addEventListener('click', quickLocal));

const uploadDialog = $('#upload-dialog');
function openUpload(source) {
  state.uploadSource = source; state.uploadFiles = [];
  $('#upload-source-name').textContent = `将文件写入“${source.name}”。保存后会自动开始 Wiki 构建。`;
  $('#file-input').value = ''; renderUploadFiles(); uploadDialog.showModal();
}
function renderUploadFiles() {
  const markup = state.uploadFiles.map((file) => `<div class="file-row"><span>${escapeHTML(file.name)}</span><span>${(file.size / 1024).toFixed(1)} KiB</span></div>`).join('');
  $('#file-list').innerHTML = markup;
  $('#source-file-list').innerHTML = markup;
}
function addFiles(files) {
  const byName = new Map(state.uploadFiles.map((file) => [file.name, file]));
  [...files].forEach((file) => byName.set(file.name, file));
  state.uploadFiles = [...byName.values()]; renderUploadFiles();
}
$('#file-input').addEventListener('change', (event) => addFiles(event.target.files));
$('#source-file-input').addEventListener('change', (event) => addFiles(event.target.files));
[$('#drop-zone'), $('#source-drop-zone')].forEach((dropZone) => {
  ['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
  dropZone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));
});

function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function uploadFilesToSource(source, selectedFiles) {
  const files = [];
  for (const file of selectedFiles) {
    files.push({
      filename: file.name,
      content_base64: await fileAsBase64(file),
      modified_at: new Date(file.lastModified).toISOString(),
    });
  }
  return api(`/api/sources/${source.id}/files`, {
    method: 'POST',
    body: JSON.stringify({ files, sync: !source.wiki_id }),
  });
}

$('#upload-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!state.uploadFiles.length) return toast('请先选择文件', 'error');
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true; button.textContent = '正在上传…';
  try {
    const result = await uploadFilesToSource(state.uploadSource, state.uploadFiles);
    uploadDialog.close();
    toast(`${result.saved.length} 个文件已保存`);
    await refreshData({ quiet: true });
    if (state.uploadSource.wiki_id) await startWikiBuild();
  } catch (error) { toast(error.message, 'error'); }
  finally { button.disabled = false; button.textContent = '保存文件'; }
});

async function copyText(value) {
  try { await navigator.clipboard.writeText(value); toast('已复制到剪贴板'); }
  catch (_) { toast('无法访问剪贴板，请手动复制', 'error'); }
}
$('#copy-mcp').addEventListener('click', () => copyText($('#mcp-url').textContent));
$('#copy-config').addEventListener('click', () => copyText($('#mcp-config').textContent));

const canvas = $('#graph-canvas');
const context = canvas.getContext('2d');
const graphViewport = { scale: 1, x: 0, y: 0, dragging: false, dragNode: null, lastX: 0, lastY: 0, ticks: 0 };

function resizeGraphCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawGraph();
}
new ResizeObserver(resizeGraphCanvas).observe(canvas);

function seededAngle(id) {
  let hash = 0;
  for (const char of id) hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0;
  return ((hash >>> 0) % 10000) / 10000 * Math.PI * 2;
}

function initializeGraph(nodes, edges) {
  const loadedFor = currentWiki()?.published_group_id || state.graph.loadedFor;
  const selectedId = state.graph.selected?.id;
  const nodeMap = new Map();
  const radius = Math.max(100, Math.sqrt(nodes.length) * 42);
  nodes.forEach((node, index) => {
    const angle = seededAngle(node.id) + index * .12;
    Object.assign(node, { x: Math.cos(angle) * radius * (.45 + (index % 7) / 10), y: Math.sin(angle) * radius * (.45 + (index % 5) / 10), vx: 0, vy: 0 });
    nodeMap.set(node.id, node);
  });
  edges.forEach((edge) => { edge.a = nodeMap.get(edge.source); edge.b = nodeMap.get(edge.target); });
  state.graph = {
    nodes,
    edges: edges.filter((edge) => edge.a && edge.b),
    selected: nodes.find((node) => node.id === selectedId) || nodes[0] || null,
    loadedFor,
  };
  graphViewport.scale = Math.min(1.25, Math.max(.4, 520 / (radius * 2)));
  graphViewport.x = canvas.clientWidth / 2; graphViewport.y = canvas.clientHeight / 2; graphViewport.ticks = 0;
  $('#graph-empty').classList.toggle('hidden', nodes.length > 0);
  $('#graph-empty').textContent = nodes.length ? '' : '当前 Wiki 还没有实体节点';
  requestAnimationFrame(simulateGraph);
}

function simulateGraph() {
  if (graphViewport.ticks < 180 && state.graph.nodes.length) {
    const nodes = state.graph.nodes;
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      a.vx += -a.x * .0007; a.vy += -a.y * .0007;
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j]; let dx = b.x - a.x; let dy = b.y - a.y;
        const distance2 = Math.max(100, dx * dx + dy * dy); const force = 72 / distance2;
        dx *= force; dy *= force; a.vx -= dx; a.vy -= dy; b.vx += dx; b.vy += dy;
      }
    }
    state.graph.edges.forEach((edge) => {
      const dx = edge.b.x - edge.a.x; const dy = edge.b.y - edge.a.y;
      const distance = Math.max(1, Math.hypot(dx, dy)); const force = (distance - 95) * .0008;
      edge.a.vx += dx * force; edge.a.vy += dy * force; edge.b.vx -= dx * force; edge.b.vy -= dy * force;
    });
    nodes.forEach((node) => { if (node !== graphViewport.dragNode) { node.vx *= .88; node.vy *= .88; node.x += node.vx; node.y += node.vy; } });
    graphViewport.ticks += 1;
  }
  drawGraph();
  if (graphViewport.ticks < 180 || graphViewport.dragging) requestAnimationFrame(simulateGraph);
}

function screenPoint(node) { return { x: node.x * graphViewport.scale + graphViewport.x, y: node.y * graphViewport.scale + graphViewport.y }; }
function worldPoint(x, y) { return { x: (x - graphViewport.x) / graphViewport.scale, y: (y - graphViewport.y) / graphViewport.scale }; }

function drawGraph() {
  const width = canvas.clientWidth; const height = canvas.clientHeight;
  context.clearRect(0, 0, width, height);
  context.lineWidth = 1;
  state.graph.edges.forEach((edge) => {
    const a = screenPoint(edge.a); const b = screenPoint(edge.b);
    context.strokeStyle = 'rgba(86,96,109,.34)';
    context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    const arrowX = b.x - Math.cos(angle) * 11; const arrowY = b.y - Math.sin(angle) * 11;
    context.beginPath(); context.moveTo(arrowX, arrowY);
    context.lineTo(arrowX - Math.cos(angle - .45) * 6, arrowY - Math.sin(angle - .45) * 6);
    context.lineTo(arrowX - Math.cos(angle + .45) * 6, arrowY - Math.sin(angle + .45) * 6);
    context.closePath(); context.fillStyle = 'rgba(86,96,109,.48)'; context.fill();
  });
  state.graph.nodes.forEach((node) => {
    const point = screenPoint(node); const selected = state.graph.selected?.id === node.id;
    const radius = selected ? 8 : 5.5;
    context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = selected ? '#556f42' : '#8da676'; context.fill();
    if (selected) { context.strokeStyle = 'rgba(85,111,66,.22)'; context.lineWidth = 7; context.stroke(); context.lineWidth = 1; }
    if (graphViewport.scale > .55 || selected) {
      context.fillStyle = '#4f5661'; context.font = `${selected ? 11 : 9}px sans-serif`;
      context.fillText(node.name.slice(0, 22), point.x + 10, point.y + 3);
    }
  });
}

function findNode(x, y) {
  return state.graph.nodes.find((node) => { const p = screenPoint(node); return Math.hypot(p.x - x, p.y - y) < 13; });
}
function showNode(node) {
  state.graph.selected = node; drawGraph();
  const relations = state.graph.edges.filter((edge) => edge.source === node.id || edge.target === node.id).slice(0, 18);
  $('#graph-inspector').innerHTML = `<h3>${escapeHTML(node.name)}</h3><p>${escapeHTML(node.summary || '暂无摘要')}</p><p>${escapeHTML(typeNames[nodeType(node)] || nodeType(node))} · ${relations.length} 条直接关系</p>`;
  renderWikiTree();
}

canvas.addEventListener('pointerdown', (event) => {
  canvas.setPointerCapture(event.pointerId); graphViewport.dragging = true; graphViewport.lastX = event.offsetX; graphViewport.lastY = event.offsetY;
  graphViewport.dragNode = findNode(event.offsetX, event.offsetY);
  if (graphViewport.dragNode) showNode(graphViewport.dragNode);
});
canvas.addEventListener('pointermove', (event) => {
  if (!graphViewport.dragging) return;
  const dx = event.offsetX - graphViewport.lastX; const dy = event.offsetY - graphViewport.lastY;
  if (graphViewport.dragNode) { const point = worldPoint(event.offsetX, event.offsetY); graphViewport.dragNode.x = point.x; graphViewport.dragNode.y = point.y; graphViewport.dragNode.vx = 0; graphViewport.dragNode.vy = 0; }
  else { graphViewport.x += dx; graphViewport.y += dy; }
  graphViewport.lastX = event.offsetX; graphViewport.lastY = event.offsetY; drawGraph();
});
canvas.addEventListener('pointerup', () => { graphViewport.dragging = false; graphViewport.dragNode = null; });
canvas.addEventListener('wheel', (event) => {
  event.preventDefault(); const before = worldPoint(event.offsetX, event.offsetY);
  graphViewport.scale = Math.min(3, Math.max(.18, graphViewport.scale * Math.exp(-event.deltaY * .001)));
  graphViewport.x = event.offsetX - before.x * graphViewport.scale; graphViewport.y = event.offsetY - before.y * graphViewport.scale; drawGraph();
}, { passive: false });

async function loadCurrentGraph() {
  const wiki = currentWiki();
  if (!wiki?.published_group_id) return toast('Wiki 尚未发布', 'error');
  $('#graph-empty').classList.remove('hidden');
  $('#graph-empty').textContent = '正在加载已发布图谱…';
  try {
    const payload = await api(`/api/wikis/${wiki.id}/graph`);
    initializeGraph(payload.nodes || [], payload.edges || []);
    renderWikiTree();
  } catch (error) {
    $('#graph-empty').textContent = error.message;
    toast(error.message, 'error');
  }
}

function resetGraphView() {
  if (!state.graph.nodes.length) return;
  initializeGraph(state.graph.nodes, state.graph.edges.map((edge) => ({ ...edge })));
}

async function startWikiBuild() {
  const wiki = currentWiki();
  if (!wiki || !state.sources.length || wiki.candidate_status === 'building') return;
  try {
    const result = await api(`/api/wikis/${wiki.id}/build`, { method: 'POST' });
    appendAssistant(`已提交 Wiki 构建任务，${result.jobs.length} 个数据源进入离线队列。完成后会自动发布。`);
    toast('Wiki 构建已开始');
    navigate('wiki');
    await refreshData({ quiet: true });
  } catch (error) {
    toast(error.message, 'error');
  }
}

function taskStatusName(status) {
  return { empty: '尚未构建', building: '构建中', ready: '已完成', failed: '失败' }[status] || status;
}

function openTaskDialog() {
  const wiki = currentWiki();
  const task = state.wikiTask;
  if (!wiki || !task) return toast('当前没有 Wiki 任务', 'error');
  const errors = (task.jobs || []).filter((job) => job.error).map((job) => job.error);
  $('#task-detail').innerHTML = `
    <div class="task-hero"><h2>Wiki ${taskStatusName(task.status)}</h2><p>${task.status === 'building' ? '任务会在后台继续运行，可关闭此窗口。' : task.status === 'failed' ? '候选版本未发布，线上继续读取上一版本。' : '当前结果已发布，Wiki 页面与 MCP 读取同一版本。'}</p></div>
    <dl class="detail-grid"><dt>任务 ID</dt><dd>${escapeHTML(task.id || '—')}</dd><dt>任务类型</dt><dd>${escapeHTML(task.type)}</dd><dt>当前状态</dt><dd>${escapeHTML(taskStatusName(task.status))}</dd><dt>开始时间</dt><dd>${formatDate(task.started_at)}</dd><dt>完成时间</dt><dd>${formatDate(task.finished_at)}</dd><dt>数据范围</dt><dd>${task.source_count} 项关联数据</dd><dt>策略版本</dt><dd>v${task.plan_version}</dd></dl>
    ${errors.length ? `<div class="form-callout">${escapeHTML(errors.join('\n'))}</div>` : ''}
    ${task.status === 'failed' ? '<div class="modal-actions"><button class="primary-button" data-task-action="retry">重新开始</button></div>' : ''}`;
  $('#task-dialog').showModal();
}

function openPlanDialog() {
  const plan = state.wikiPlan;
  if (!plan) return toast('当前没有 Wiki 计划', 'error');
  const content = plan.plan || {};
  $('#plan-detail').innerHTML = `
    <h2>${escapeHTML(content.template_name || '项目 Wiki')} · v${plan.version}</h2>
    <div class="form-callout"><strong>创建目标</strong><br>${escapeHTML(plan.goal || '未填写')}</div>
    <div class="plan-types">${(content.entity_types || []).map((type) => `<div class="plan-type"><strong>${escapeHTML(type.name)}</strong><small>${escapeHTML((type.fields || []).join(' · '))}</small></div>`).join('')}</div>
    <h3>Link 类型</h3><p>${escapeHTML((content.link_types || []).join('、'))}</p>
    <h3>质量规则</h3><ul>${(content.quality_rules || []).map((rule) => `<li>${escapeHTML(rule)}</li>`).join('')}</ul>`;
  $('#plan-dialog').showModal();
}

function openCopilot() {
  $('#copilot').classList.remove('closed');
  $('#copilot-fab').classList.add('hidden');
  $('#chat-input').focus();
}

function appendUser(text) {
  $('#chat-log').insertAdjacentHTML('beforeend', `<div class="user-message">${escapeHTML(text)}</div>`);
  $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
}

function appendAssistant(text, card = '') {
  $('#chat-log').insertAdjacentHTML('beforeend', `<div class="assistant-message">${escapeHTML(text)}</div>${card}`);
  $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
}

function extractWikiName(text) {
  const quoted = text.match(/[“"]([^”"]+)[”"]/);
  if (quoted) return quoted[1].trim();
  const colon = text.split(/[：:]/);
  if (colon.length > 1) return colon.at(-1).replace(/wiki/ig, '').trim();
  const match = text.match(/(?:创建|新建)(?:一个|新的|一份)*\s*([^，。]{2,40}?)\s*(?:Wiki|wiki)/);
  if (match && !/项目$/.test(match[1])) return match[1].trim();
  return '';
}

function startCreateFlow(seed = '', afterCreate = null) {
  state.createFlow = {
    step: 'name', name: '', goal: '', dataScope: 'specified', afterCreate,
  };
  openCopilot();
  const name = extractWikiName(seed);
  if (name) {
    state.createFlow.name = name;
    state.createFlow.step = 'goal';
    appendAssistant(`好的，新 Wiki 名称是“${name}”。请说明建库目标：希望管理哪些对象、给谁使用、主要回答什么问题？`);
  } else if (afterCreate === 'add-data') {
    appendAssistant('添加数据前，需要先确定它属于哪个 Wiki。我们先创建一个项目 Wiki，请告诉我名称。');
  } else appendAssistant('我们来创建一个项目 Wiki。先告诉我 Wiki 名称。');
}

function showPlanConfirmation() {
  const flow = state.createFlow;
  const card = `<div class="chat-card"><h4>${escapeHTML(flow.name)} · 创建计划 v1</h4><p>${escapeHTML(flow.goal)}</p><p>模板：项目 Wiki<br>数据范围：${flow.dataScope === 'all' ? '全部数据' : '指定数据'}<br>实体类型：项目、产品、产品模块、产品功能、版本、产品需求、缺陷、人员</p><button class="primary-button" data-chat-action="confirm-create">确认计划</button><button class="quiet-button" data-chat-action="cancel-create">重新填写</button></div>`;
  appendAssistant('我已根据目标生成 MVP 项目 Wiki 计划。确认后创建实例，再添加数据。', card);
}

async function confirmCreateFlow() {
  const flow = state.createFlow;
  if (!flow) return;
  const afterCreate = flow.afterCreate;
  try {
    const wiki = await api('/api/wikis', {
      method: 'POST',
      body: JSON.stringify({ name: flow.name, goal: flow.goal, data_scope: flow.dataScope }),
    });
    state.currentWikiId = wiki.id;
    localStorage.setItem('vaka.currentWikiId', wiki.id);
    state.createFlow = null;
    appendAssistant(`“${wiki.name}”已创建，策略 v1 已锁定。现在添加本地文件、飞书或 MeeGo 数据。`, '<div class="chat-card"><button class="primary-button" data-chat-action="add-data">添加数据</button><button class="quiet-button" data-chat-action="view-plan">查看计划</button></div>');
    await refreshData();
    if (afterCreate === 'add-data') {
      navigate('library');
      await openSourceDialog();
    }
  } catch (error) {
    appendAssistant(`创建失败：${error.message}`);
  }
}

async function handleChat(text) {
  const flow = state.createFlow;
  if (flow) {
    if (flow.step === 'name') {
      const name = extractWikiName(text) || text.replace(/wiki/ig, '').trim();
      if (!name) return appendAssistant('请提供一个明确的 Wiki 名称。');
      flow.name = name.slice(0, 120); flow.step = 'goal';
      return appendAssistant(`名称已记录为“${flow.name}”。请继续说明建库目标和主要使用场景。`);
    }
    if (flow.step === 'goal') {
      flow.goal = text; flow.step = 'scope';
      return appendAssistant('数据范围使用“全部数据”，还是只选择指定文件、目录或 MeeGo 项目？');
    }
    if (flow.step === 'scope') {
      flow.dataScope = text.includes('全部') ? 'all' : 'specified'; flow.step = 'confirm';
      return showPlanConfirmation();
    }
  }
  if (/创建|新建/.test(text) && /wiki/i.test(text)) return startCreateFlow(text);
  if (/任务|进度|构建/.test(text)) {
    if (!currentWiki()) return appendAssistant('当前还没有 Wiki。你可以先说“帮我创建一个 Wiki”。');
    openTaskDialog();
    return appendAssistant(`当前任务状态：${taskStatusName(state.wikiTask?.status || 'empty')}。`);
  }
  if (/计划|策略/.test(text)) {
    openPlanDialog(); return appendAssistant('已打开当前 Wiki 的只读创建计划。');
  }
  const wiki = currentWiki();
  if (!wiki?.published_group_id) return appendAssistant('当前 Wiki 还没有已发布内容。先添加数据，完成构建后我就能检索。');
  try {
    appendAssistant('正在检索当前 Wiki 的已发布事实…');
    const payload = await api(`/api/wikis/${wiki.id}/search`, { method: 'POST', body: JSON.stringify({ query: text, max_facts: 8 }) });
    const card = payload.facts.length ? `<div class="chat-card">${payload.facts.map((fact) => `<p><strong>${escapeHTML(fact.name || '事实')}</strong><br>${escapeHTML(fact.fact)}</p>`).join('')}</div>` : '';
    appendAssistant(payload.facts.length ? `找到 ${payload.facts.length} 条相关事实。` : '没有找到足够相关的已发布事实。', card);
  } catch (error) { appendAssistant(`检索失败：${error.message}`); }
}

window.addEventListener('hashchange', () => navigate(location.hash.slice(1) in pageLabels ? location.hash.slice(1) : 'wiki'));

async function boot() {
  const initial = location.hash.slice(1);
  navigate(pageLabels[initial] ? initial : 'wiki');
  await refreshData();
  await loadConnections({ quiet: true });
  setInterval(() => refreshData({ quiet: true }), 5000);
}

$('#wiki-select').addEventListener('change', async (event) => {
  state.currentWikiId = event.target.value || null;
  if (state.currentWikiId) localStorage.setItem('vaka.currentWikiId', state.currentWikiId);
  await refreshData();
});

$('#new-wiki-button').addEventListener('click', () => startCreateFlow());
$('#task-link').addEventListener('click', openTaskDialog);
$('#build-task-button').addEventListener('click', openTaskDialog);
$('#plan-link').addEventListener('click', openPlanDialog);
$('#mcp-link').addEventListener('click', () => currentWiki() ? $('#mcp-dialog').showModal() : toast('请先创建 Wiki', 'error'));
$('#manage-data-button').addEventListener('click', () => navigate('library'));
$('#graph-load').addEventListener('click', loadCurrentGraph);
$('#graph-reset').addEventListener('click', resetGraphView);
$('#task-detail').addEventListener('click', (event) => {
  if (event.target.closest('[data-task-action="retry"]')) { $('#task-dialog').close(); startWikiBuild(); }
});
$('#wiki-tree').addEventListener('click', (event) => {
  const button = event.target.closest('[data-node-id]');
  if (!button) return;
  state.graph.selected = state.graph.nodes.find((node) => node.id === button.dataset.nodeId) || null;
  state.wikiMode = 'directory';
  renderWiki();
});
$$('[data-wiki-mode]').forEach((button) => button.addEventListener('click', () => {
  state.wikiMode = button.dataset.wikiMode;
  $$('[data-wiki-mode]').forEach((item) => item.classList.toggle('active', item === button));
  renderWiki();
}));
$$('[data-entity-tab]').forEach((button) => button.addEventListener('click', () => {
  state.entityTab = button.dataset.entityTab;
  $$('[data-entity-tab]').forEach((item) => item.classList.toggle('active', item === button));
  renderEntityBody();
}));
document.addEventListener('click', (event) => {
  const action = event.target.closest('[data-action]')?.dataset.action;
  if (action === 'start-create') startCreateFlow();
  if (action === 'start-create-upload') startCreateFlow('', 'add-data');
  if (action === 'add-data') {
    navigate('library');
    openSourceDialog();
  }
  if (action === 'start-build') startWikiBuild();
});
$('#copilot-toggle').addEventListener('click', () => { $('#copilot').classList.add('closed'); $('#copilot-fab').classList.remove('hidden'); });
$('#copilot-fab').addEventListener('click', openCopilot);
$('#chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = $('#chat-input').value.trim();
  if (!text) return;
  $('#chat-input').value = '';
  appendUser(text);
  await handleChat(text);
});
$('#chat-suggestions').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  $('#chat-input').value = button.textContent;
  $('#chat-form').requestSubmit();
});
$('#chat-log').addEventListener('click', (event) => {
  const action = event.target.closest('[data-chat-action]')?.dataset.chatAction;
  if (action === 'confirm-create') confirmCreateFlow();
  if (action === 'cancel-create') startCreateFlow('', state.createFlow?.afterCreate || null);
  if (action === 'add-data') { navigate('library'); openSourceDialog(); }
  if (action === 'view-plan') openPlanDialog();
});
boot();
