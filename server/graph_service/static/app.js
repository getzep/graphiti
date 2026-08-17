const state = {
  status: null,
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
      connectionId: null, parentId: '', path: [], items: [], nextPage: null,
      project: null, workItemTypes: new Map(),
    },
  },
  sourceFilter: 'all',
  uploadSource: null,
  uploadFiles: [],
  graph: { nodes: [], edges: [], selected: null },
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
  return state.connections.filter((connection) => connectionProvider(connection) === provider);
}

function selectedConnection(provider) {
  const id = state.connectionSelection[provider];
  return connectionsFor(provider).find((connection) => connection.id === id) || null;
}

const pageLabels = {
  overview: '知识运行总览', sources: '数据源管理', graph: '图谱浏览', search: '搜索与 Agent',
};

function navigate(view) {
  $$('.view').forEach((element) => element.classList.toggle('active', element.id === `view-${view}`));
  $$('.nav-item').forEach((element) => element.classList.toggle('active', element.dataset.nav === view));
  $('#page-label').textContent = pageLabels[view];
  location.hash = view;
  $('.sidebar').classList.remove('open');
  if (view === 'graph') setTimeout(resizeGraphCanvas, 20);
}

document.addEventListener('click', (event) => {
  const nav = event.target.closest('[data-nav]');
  if (nav) {
    event.preventDefault();
    navigate(nav.dataset.nav);
  }
});

$('#mobile-menu').addEventListener('click', () => $('.sidebar').classList.toggle('open'));
$$('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));

function renderStatus() {
  if (!state.status) return;
  const { database, llm, embedding, connectors, stats } = state.status;
  $('#system-chips').innerHTML = `
    <span class="chip"><i class="status-dot ${database.ready === false ? 'error' : ''}"></i>${escapeHTML(database.provider)} ${database.ready === false ? '未连接' : '已连接'}</span>
    <span class="chip"><i class="status-dot ${llm.configured ? '' : 'error'}"></i>${llm.configured ? '模型已配置' : '模型待配置'}</span>
  `;
  const metrics = $$('#metric-strip article');
  metrics[0].querySelector('strong').textContent = stats.sources;
  metrics[1].querySelector('strong').textContent = stats.items;
  metrics[2].querySelector('strong').textContent = stats.active_jobs;
  metrics[3].querySelector('strong').textContent = embedding.provider === 'local_hash' ? 'LOCAL HASH' : 'REMOTE MODEL';
  $('#readiness-list').innerHTML = [
    ['Neo4j / Graph DB', database.ready === false ? '未连通' : database.configured ? 'READY' : 'MISSING', database.ready !== false && database.configured],
    ['火山方舟 / LLM', llm.configured ? (llm.model || 'READY') : 'MISSING', llm.configured],
    ['本地文件', 'READY', true],
    ['飞书连接器', connectors.feishu ? 'READY' : 'ENV NEEDED', connectors.feishu],
    ['MeeGo 连接器', connectors.meego ? 'READY' : 'ENV NEEDED', connectors.meego],
  ].map(([label, value, ready]) => `<div class="readiness-row"><span>${escapeHTML(label)}</span><span class="${ready ? '' : 'offline'}">${escapeHTML(value)}</span></div>`).join('');
  $('#mcp-url').textContent = state.status.mcp_url;
  $('#mcp-config').textContent = JSON.stringify({
    mcpServers: { graphiti: { url: state.status.mcp_url } },
  }, null, 2);
  const suggestedGroup = state.status.suggested_group_id || 'neo4j';
  ['#graph-group', '#search-group', '#source-form input[name="group_id"]'].forEach((selector) => {
    const input = $(selector);
    if (input && (!input.dataset.initialized || input.value === 'neo4j')) input.value = suggestedGroup;
    if (input) input.dataset.initialized = 'true';
  });
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
  return `${account}项目 ${source.config.project_key || '—'} · ${(source.config.work_item_type_keys || []).join(', ') || '自动发现工作项类型'}`;
}

function renderSources() {
  const filtered = state.sources.filter((source) => state.sourceFilter === 'all' || source.kind === state.sourceFilter);
  const grid = $('#source-grid');
  if (!filtered.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1">还没有这个类型的数据源。点击“添加数据源”开始。</div>';
    return;
  }
  grid.innerHTML = filtered.map((source) => `
    <article class="source-card" data-kind="${source.kind}" data-source-id="${source.id}">
      <div class="source-card-head"><span class="source-icon">${source.kind === 'local' ? 'FILE' : source.kind === 'feishu' ? 'LARK' : 'MEE'}</span><span class="source-status ${source.status === 'error' ? 'error' : ''}">${source.status === 'syncing' ? 'SYNCING' : source.status === 'error' ? 'ERROR' : source.enabled ? 'ACTIVE' : 'PAUSED'}</span></div>
      <h3>${escapeHTML(source.name)}</h3>
      <div class="source-meta">${escapeHTML(kindName(source.kind))} / ${escapeHTML(source.group_id)}</div>
      <p class="source-detail">${escapeHTML(sourceDetail(source))}</p>
      <div class="source-meta">上次同步 ${formatDate(source.last_sync_at)}${source.last_error ? ` · ${escapeHTML(source.last_error)}` : ''}</div>
      <div class="source-actions">
        ${source.kind === 'local' ? '<button class="primary" data-source-action="upload">上传文件</button>' : ''}
        <button class="primary" data-source-action="sync">立即同步</button>
        <button data-source-action="full-sync">全量对账</button>
        <button data-source-action="toggle">${source.enabled ? '停用' : '启用'}</button>
        <button class="danger" data-source-action="delete">删除定义</button>
      </div>
    </article>
  `).join('');
}

function renderJobs() {
  const names = Object.fromEntries(state.sources.map((source) => [source.id, source.name]));
  const rows = state.jobs.map((job) => `
    <tr>
      <td><span class="job-badge ${job.status}">${escapeHTML(job.status.toUpperCase())}</span></td>
      <td>${escapeHTML(names[job.source_id] || job.source_id.slice(0, 8))}</td>
      <td>${job.scanned}</td><td>${job.created}</td><td>${job.updated}</td><td>${job.skipped}</td>
      <td>${formatDate(job.started_at || job.created_at)}</td>
      <td title="${escapeHTML(job.error || (job.warnings || []).join('\n'))}">${escapeHTML(job.error || (job.warnings || [])[0] || '—')}</td>
    </tr>
  `).join('');
  $('#jobs-table').innerHTML = rows || '<tr><td colspan="8" class="empty-state">暂无任务</td></tr>';
  const recent = state.jobs.slice(0, 5);
  $('#recent-jobs').classList.toggle('empty-state', recent.length === 0);
  $('#recent-jobs').innerHTML = recent.length ? recent.map((job) => `
    <div class="timeline-item"><span class="job-dot ${job.status}"></span><div><strong>${escapeHTML(names[job.source_id] || '已删除的数据源')}</strong><p>${job.scanned} 扫描 · ${job.created} 新增 · ${job.updated} 更新 · ${job.skipped} 跳过</p></div><time>${formatDate(job.started_at || job.created_at)}</time></div>
  `).join('') : '还没有同步任务';
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
      connectionId, parentId: '', path: [], items: [], nextPage: null,
      loaded: false, loading: false, error: null,
      project: null, workItemTypes: new Map(),
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
    if (connectButton) {
      connectButton.disabled = !providerAvailable;
      connectButton.title = providerAvailable ? '' : '需要管理员先配置 OAuth 应用';
      connectButton.textContent = providerAvailable
        ? `连接 ${kindName(provider)}`
        : `${kindName(provider)}待管理员配置`;
    }
    const connections = connectionsFor(provider);
    const select = $(`#${provider}-connection`);
    let selectedId = state.connectionSelection[provider];
    if (!connections.some((connection) => connection.id === selectedId)) {
      selectedId = connections[0]?.id || null;
      state.connectionSelection[provider] = selectedId;
    }
    select.disabled = connections.length === 0;
    select.innerHTML = connections.length
      ? connections.map((connection) => `<option value="${escapeHTML(connection.id)}">${escapeHTML(connectionLabel(connection) + connectionStatus(connection))}</option>`).join('')
      : '<option value="">尚未连接账号</option>';
    select.value = selectedId || '';
    const connection = selectedConnection(provider);
    $(`#${provider}-connection-label`).textContent = connection
      ? connectionLabel(connection) + connectionStatus(connection)
      : '尚未连接';
    const browse = $(`[data-resource-load="${provider}"]`);
    if (browse) browse.disabled = !connection;
    const picker = state.resourcePickers[provider];
    if (picker.connectionId !== selectedId) resetResourcePicker(provider);
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

function isMeeGoProject(item, picker) {
  return picker.path.length === 0 || item.type === 'project' || Boolean(item.metadata?.project_key);
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
  } else {
    const project = picker.project
      ? `<span class="selection-chip project">项目 · ${escapeHTML(picker.project.name)}</span>` : '';
    const types = [...picker.workItemTypes.values()].map((item) => `<span class="selection-chip">${escapeHTML(item.name)}</span>`).join('');
    container.innerHTML = project
      ? `<div class="selection-chips">${project}${types}</div><button type="button" data-clear-selection="meego">清除选择</button>`
      : '尚未选择项目';
  }
}

function resourceRow(provider, item, index) {
  const picker = state.resourcePickers[provider];
  const folder = provider === 'feishu' && isFeishuFolder(item);
  const project = provider === 'meego' && isMeeGoProject(item, picker);
  let selectAction = '';
  let selected = false;
  if (provider === 'feishu' && folder && item.selectable) {
    selected = picker.folder?.id === item.id;
    selectAction = `<button type="button" data-select-folder="${index}">${selected ? '已选择' : '选择目录'}</button>`;
  } else if (provider === 'feishu' && !folder && item.selectable) {
    selected = picker.documents.has(item.id);
    selectAction = `<button type="button" data-toggle-document="${index}">${selected ? '取消' : '选择'}</button>`;
  } else if (provider === 'meego' && project && item.selectable) {
    selected = picker.project?.id === item.id;
    selectAction = `<button type="button" data-select-project="${index}">${selected ? '已选择' : '选择项目'}</button>`;
  } else if (provider === 'meego' && !project && item.selectable) {
    selected = picker.workItemTypes.has(itemKey(item, 'work_item_type'));
    selectAction = `<button type="button" data-toggle-work-type="${index}">${selected ? '取消' : '选择'}</button>`;
  }
  const openAction = item.has_children
    ? `<button type="button" data-resource-open="${index}">打开</button>` : '';
  return `<div class="resource-row ${selected ? 'selected' : ''}">
    <span class="resource-kind">${escapeHTML(resourceTypeName(item.type))}</span>
    <div><strong>${escapeHTML(item.name)}</strong><small>${item.has_children ? '包含下级资源' : escapeHTML(item.metadata?.description || '')}</small></div>
    <div class="resource-row-actions">${selectAction}${openAction}</div>
  </div>`;
}

function renderResourcePicker(provider) {
  const picker = state.resourcePickers[provider];
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

function selectMeeGoProject(item) {
  const picker = state.resourcePickers.meego;
  if (picker.project?.id !== item.id) picker.workItemTypes.clear();
  picker.project = { ...item, key: itemKey(item, 'project') };
  renderResourcePicker('meego');
}

function toggleMeeGoWorkItemType(item) {
  const picker = state.resourcePickers.meego;
  const key = itemKey(item, 'work_item_type');
  if (picker.workItemTypes.has(key)) picker.workItemTypes.delete(key);
  else picker.workItemTypes.set(key, { ...item, key });
  renderResourcePicker('meego');
}

async function refreshData({ quiet = false } = {}) {
  if (refreshData.inFlight) return refreshData.inFlight;
  refreshData.inFlight = (async () => {
    try {
      const [status, sources, jobs] = await Promise.all([
        api('/api/status'), api('/api/sources'), api('/api/jobs?limit=50'),
      ]);
      state.status = status;
      state.sources = sources;
      state.jobs = jobs;
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
$('#refresh-sources').addEventListener('click', () => refreshData());

const sourceDialog = $('#source-dialog');
const sourceForm = $('#source-form');

function setSourceKind(kind) {
  $$('.kind-fields', sourceForm).forEach((fields) => {
    fields.classList.toggle('hidden', fields.dataset.kindFields !== kind);
  });
}

async function openSourceDialog() {
  sourceForm.reset();
  setSourceKind('local');
  const group = $('input[name="group_id"]', sourceForm);
  group.value = state.status?.suggested_group_id || 'neo4j';
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

function openOAuth(provider) {
  if (!['feishu', 'meego'].includes(provider)) return;
  if (state.status?.oauth?.providers?.[provider] === false) {
    return toast(`${kindName(provider)} OAuth 需要管理员先配置应用`, 'error');
  }
  const current = state.oauthPopups.get(provider);
  if (current && !current.closed) {
    current.focus();
    return;
  }
  const width = 640; const height = 760;
  const left = Math.max(0, window.screenX + (window.outerWidth - width) / 2);
  const top = Math.max(0, window.screenY + (window.outerHeight - height) / 2);
  const popup = window.open(
    `/api/oauth/${encodeURIComponent(provider)}/start`,
    `graphiti-oauth-${provider}`,
    `popup=yes,width=${width},height=${height},left=${Math.round(left)},top=${Math.round(top)}`,
  );
  if (!popup) return toast('浏览器阻止了授权窗口，请允许本站打开弹窗后重试', 'error');
  state.oauthPopups.set(provider, popup);
  popup.focus();
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
  await loadConnections({
    preferredProvider: payload.provider,
    preferredConnectionId: payload.connection_id,
  });
  resetResourcePicker(payload.provider);
  toast(`${kindName(payload.provider)}账号已连接`);
  await loadResources(payload.provider);
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
    if (provider === 'meego' && isMeeGoProject(item, picker) && item.selectable) {
      selectMeeGoProject(item);
    }
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
  } else if (target.hasAttribute('data-select-project')) {
    const item = picker.items[Number(target.dataset.selectProject)];
    if (item) selectMeeGoProject(item);
  } else if (target.hasAttribute('data-toggle-work-type')) {
    const item = picker.items[Number(target.dataset.toggleWorkType)];
    if (item) toggleMeeGoWorkItemType(item);
  } else if (target.hasAttribute('data-clear-selection')) {
    if (provider === 'feishu') {
      picker.root = false; picker.folder = null; picker.documents.clear();
    } else {
      picker.project = null; picker.workItemTypes.clear();
    }
    renderResourcePicker(provider);
  }
});

$('#source-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const kind = form.get('kind');
  const config = {};
  const payload = {
    kind,
    name: String(form.get('name') || '').trim(),
    group_id: String(form.get('group_id') || '').trim(),
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
    if (!picker.project) return toast('请选择一个 MeeGo 项目', 'error');
    payload.connection_id = connectionId;
    config.project_key = picker.project.key;
    config.work_item_type_keys = [...picker.workItemTypes.keys()];
    config.page_size = Number(form.get('page_size') || 100);
  }
  try {
    const source = await api('/api/sources', {
      method: 'POST', body: JSON.stringify(payload),
    });
    sourceDialog.close(); event.currentTarget.reset();
    setSourceKind('local');
    resetResourcePicker('feishu'); resetResourcePicker('meego');
    toast(`已创建“${source.name}”`);
    await refreshData({ quiet: true });
    if (source.kind === 'local') openUpload(source);
  } catch (error) { toast(error.message, 'error'); }
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
  let source = state.sources.find((item) => item.kind === 'local' && item.enabled);
  if (!source) {
    try {
      source = await api('/api/sources', { method: 'POST', body: JSON.stringify({ kind: 'local', name: '本地资料库', group_id: state.status?.suggested_group_id || 'neo4j', config: {} }) });
      await refreshData({ quiet: true });
    } catch (error) { return toast(error.message, 'error'); }
  }
  openUpload(source);
}
$$('[data-action="quick-local"]').forEach((button) => button.addEventListener('click', quickLocal));

const uploadDialog = $('#upload-dialog');
function openUpload(source) {
  state.uploadSource = source; state.uploadFiles = [];
  $('#upload-source-name').textContent = `将文件写入“${source.name}”，Group ID：${source.group_id}`;
  $('#file-input').value = ''; renderUploadFiles(); uploadDialog.showModal();
}
function renderUploadFiles() {
  $('#file-list').innerHTML = state.uploadFiles.map((file) => `<div class="file-row"><span>${escapeHTML(file.name)}</span><span>${(file.size / 1024).toFixed(1)} KiB</span></div>`).join('');
}
function addFiles(files) {
  const byName = new Map(state.uploadFiles.map((file) => [file.name, file]));
  [...files].forEach((file) => byName.set(file.name, file));
  state.uploadFiles = [...byName.values()]; renderUploadFiles();
}
$('#file-input').addEventListener('change', (event) => addFiles(event.target.files));
const dropZone = $('#drop-zone');
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));

function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
$('#upload-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!state.uploadFiles.length) return toast('请先选择文件', 'error');
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true; button.textContent = '正在上传…';
  try {
    const files = [];
    for (const file of state.uploadFiles) {
      files.push({
        filename: file.name,
        content_base64: await fileAsBase64(file),
        modified_at: new Date(file.lastModified).toISOString(),
      });
    }
    const result = await api(`/api/sources/${state.uploadSource.id}/files`, { method: 'POST', body: JSON.stringify({ files, sync: true }) });
    uploadDialog.close();
    toast(`${result.saved.length} 个文件已保存，同步任务已启动`);
    await refreshData({ quiet: true });
  } catch (error) { toast(error.message, 'error'); }
  finally { button.disabled = false; button.textContent = '上传并同步'; }
});

$('#search-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const container = $('#search-results');
  container.className = 'search-results empty-state'; container.textContent = '正在检索图谱…';
  try {
    const payload = await api('/search', { method: 'POST', body: JSON.stringify({ query: $('#search-query').value, group_ids: [$('#search-group').value], max_facts: 12 }) });
    container.className = 'search-results';
    container.innerHTML = payload.facts.length ? payload.facts.map((fact) => `
      <article class="fact-result"><strong>${escapeHTML(fact.name || 'FACT')}</strong><p>${escapeHTML(fact.fact)}</p><small>有效：${formatDate(fact.valid_at)} · ${fact.invalid_at ? `失效：${formatDate(fact.invalid_at)}` : '当前有效'}</small></article>
    `).join('') : '<div class="empty-state">没有找到相关事实</div>';
  } catch (error) { container.textContent = error.message; toast(error.message, 'error'); }
});

async function copyText(value) {
  try { await navigator.clipboard.writeText(value); toast('已复制到剪贴板'); }
  catch (_) { toast('无法访问剪贴板，请手动复制', 'error'); }
}
$('#copy-mcp').addEventListener('click', () => copyText($('#mcp-url').textContent));
$('#copy-mcp-mini').addEventListener('click', () => copyText(state.status?.mcp_url || 'http://localhost:8001/mcp/'));
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
  const nodeMap = new Map();
  const radius = Math.max(100, Math.sqrt(nodes.length) * 42);
  nodes.forEach((node, index) => {
    const angle = seededAngle(node.id) + index * .12;
    Object.assign(node, { x: Math.cos(angle) * radius * (.45 + (index % 7) / 10), y: Math.sin(angle) * radius * (.45 + (index % 5) / 10), vx: 0, vy: 0 });
    nodeMap.set(node.id, node);
  });
  edges.forEach((edge) => { edge.a = nodeMap.get(edge.source); edge.b = nodeMap.get(edge.target); });
  state.graph = { nodes, edges: edges.filter((edge) => edge.a && edge.b), selected: null };
  graphViewport.scale = Math.min(1.25, Math.max(.4, 520 / (radius * 2)));
  graphViewport.x = canvas.clientWidth / 2; graphViewport.y = canvas.clientHeight / 2; graphViewport.ticks = 0;
  $('#graph-empty').classList.toggle('hidden', nodes.length > 0);
  $('#graph-empty').textContent = nodes.length ? '' : '当前 Group 还没有实体节点';
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
    context.strokeStyle = edge.expired_at ? 'rgba(255,118,94,.20)' : 'rgba(114,132,142,.32)';
    context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
  });
  state.graph.nodes.forEach((node) => {
    const point = screenPoint(node); const selected = state.graph.selected?.id === node.id;
    const radius = selected ? 8 : 5.5;
    context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = selected ? '#ff765e' : '#c8ff4d'; context.fill();
    if (selected) { context.strokeStyle = 'rgba(255,118,94,.3)'; context.lineWidth = 7; context.stroke(); context.lineWidth = 1; }
    if (graphViewport.scale > .55 || selected) {
      context.fillStyle = selected ? '#f3f0e8' : 'rgba(214,220,222,.72)'; context.font = `${selected ? 11 : 9}px sans-serif`;
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
  $('#graph-inspector').innerHTML = `<span class="eyebrow">ENTITY</span><h2>${escapeHTML(node.name)}</h2><p>${escapeHTML(node.summary || '暂无摘要')}</p><div class="source-meta">${escapeHTML((node.labels || []).join(' · '))}</div><div class="relation-list">${relations.map((edge) => `<div class="relation-chip"><strong>${escapeHTML(edge.name || 'RELATES_TO')}</strong><br>${escapeHTML(edge.fact || '')}</div>`).join('') || '<p>当前快照中没有可见关系</p>'}</div>`;
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

$('#graph-form').addEventListener('submit', async (event) => {
  event.preventDefault(); $('#graph-empty').classList.remove('hidden'); $('#graph-empty').textContent = '正在从 Neo4j 加载…';
  try {
    const payload = await api(`/api/graph?group_id=${encodeURIComponent($('#graph-group').value)}&limit=120`);
    initializeGraph(payload.nodes, payload.edges);
  } catch (error) { $('#graph-empty').textContent = error.message; toast(error.message, 'error'); }
});

window.addEventListener('hashchange', () => navigate(location.hash.slice(1) in pageLabels ? location.hash.slice(1) : 'overview'));

async function boot() {
  const initial = location.hash.slice(1);
  navigate(pageLabels[initial] ? initial : 'overview');
  await refreshData();
  await loadConnections({ quiet: true });
  setInterval(() => refreshData({ quiet: true }), 5000);
}
boot();
