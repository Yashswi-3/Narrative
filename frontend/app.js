const API_BASE = "http://localhost:8000";

const elements = {
  healthDot: document.getElementById("healthDot"),
  healthText: document.getElementById("healthText"),
  queryInput: document.getElementById("queryInput"),
  timeFilter: document.getElementById("timeFilter"),
  limitSelect: document.getElementById("limitSelect"),
  searchButton: document.getElementById("searchButton"),
  cancelButton: document.getElementById("cancelButton"),
  searchError: document.getElementById("searchError"),
  progressSection: document.getElementById("progressSection"),
  resultsSection: document.getElementById("resultsSection"),
  stageRows: Array.from(document.querySelectorAll(".stage-row")),
  resultQuery: document.getElementById("resultQuery"),
  summaryBadges: document.getElementById("summaryBadges"),
  summaryText: document.getElementById("summaryText"),
  resultTimestamp: document.getElementById("resultTimestamp"),
  copySummaryButton: document.getElementById("copySummaryButton"),
  exportJsonButton: document.getElementById("exportJsonButton"),
  topicsContainer: document.getElementById("topicsContainer"),
  entitiesContainer: document.getElementById("entitiesContainer"),
  togglePostsButton: document.getElementById("togglePostsButton"),
  topPostsContainer: document.getElementById("topPostsContainer"),
  historyList: document.getElementById("historyList"),
  historyEmpty: document.getElementById("historyEmpty"),
  clearHistoryButton: document.getElementById("clearHistoryButton"),
  newsPanel: document.getElementById("newsPanel")
};

let currentEventSource = null;
let currentResult = null;
let historyCache = [];
let activeStage = 0;
let stageStartTimes = {};
let searchDebounceTimer = null;
let requestInFlight = false;
let healthPollTimer = null;
let fetchAbortController = null;

function showInlineError(message) {
  elements.searchError.textContent = message;
  elements.searchError.classList.remove("hidden");
}

function clearInlineError() {
  elements.searchError.textContent = "";
  elements.searchError.classList.add("hidden");
}

function setSearchEnabled(enabled) {
  elements.searchButton.disabled = !enabled;
}

function resetStages() {
  activeStage = 0;
  stageStartTimes = {};

  elements.stageRows.forEach((row) => {
    row.classList.remove("active", "done");
    row.querySelector(".stage-message").textContent = "Waiting...";
    row.querySelector(".stage-state").textContent = "waiting";
    row.querySelector(".stage-time").textContent = "";
    row.querySelector(".stage-badge").textContent = row.dataset.stage;
  });
}

function formatDuration(ms) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function markStage(stage, message, isDone = false) {
  const now = performance.now();

  if (activeStage && stage !== activeStage) {
    const previousRow = elements.stageRows.find((row) => Number(row.dataset.stage) === activeStage);
    if (previousRow && stageStartTimes[activeStage]) {
      const elapsed = now - stageStartTimes[activeStage];
      previousRow.classList.remove("active");
      previousRow.classList.add("done");
      previousRow.querySelector(".stage-state").textContent = "done";
      previousRow.querySelector(".stage-time").textContent = formatDuration(elapsed);
      previousRow.querySelector(".stage-badge").textContent = "✓";
    }
  }

  const row = elements.stageRows.find((item) => Number(item.dataset.stage) === stage);
  if (!row) {
    return;
  }

  if (!stageStartTimes[stage]) {
    stageStartTimes[stage] = now;
  }

  activeStage = stage;
  row.classList.add("active");
  row.querySelector(".stage-message").textContent = message;
  row.querySelector(".stage-state").textContent = isDone ? "done" : "active";

  if (isDone) {
    const elapsed = now - stageStartTimes[stage];
    row.classList.remove("active");
    row.classList.add("done");
    row.querySelector(".stage-time").textContent = formatDuration(elapsed);
    row.querySelector(".stage-badge").textContent = "✓";
  }
}

function resetSearchUI() {
  requestInFlight = false;
  setSearchEnabled(true);
}

function closeEventSource() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
}

function cancelSearch() {
  closeEventSource();
  if (fetchAbortController) {
    fetchAbortController.abort();
    fetchAbortController = null;
  }
  resetStages();
  elements.progressSection.classList.add("hidden");
  resetSearchUI();
}

function createBadge(text, className = "") {
  const badge = document.createElement("span");
  badge.className = `badge ${className}`.trim();
  badge.textContent = text;
  return badge;
}

function renderEntities(result) {
  const groups = [
    { key: "PERSON", label: "People", color: "purple" },
    { key: "ORG", label: "Organizations", color: "blue" },
    { key: "GPE", label: "Locations", color: "green" },
    { key: "EVENT", label: "Events", color: "yellow" }
  ];

  elements.entitiesContainer.innerHTML = "";

  groups.forEach((group) => {
    const values = (result.key_entities && result.key_entities[group.key]) || [];
    if (!values.length) {
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "entity-group";

    const title = document.createElement("h4");
    title.textContent = group.label;
    wrapper.appendChild(title);

    values.forEach((item) => {
      const chip = document.createElement("span");
      chip.className = `entity-chip ${group.color}`;
      chip.textContent = item;
      wrapper.appendChild(chip);
    });

    elements.entitiesContainer.appendChild(wrapper);
  });
}

function buildHeadlineRow(headlineData) {
  const row = document.createElement("div");
  row.className = "news-headline-row";

  const publisher = document.createElement("span");
  publisher.className = "news-publisher";
  publisher.textContent = headlineData.publisher || "Unknown";

  const headline = document.createElement("p");
  headline.className = "news-headline-text";
  headline.textContent = headlineData.headline || "Untitled headline";

  const metaRow = document.createElement("div");
  metaRow.className = "news-meta-row";

  const timeBadge = document.createElement("span");
  timeBadge.className = "news-time-badge";
  timeBadge.textContent = headlineData.time_ago || "recently";

  const link = document.createElement("a");
  link.className = "news-link";
  link.href = headlineData.url || "#";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "↗ Read";

  metaRow.appendChild(timeBadge);
  metaRow.appendChild(link);

  row.appendChild(publisher);
  row.appendChild(headline);
  row.appendChild(metaRow);

  return row;
}

function renderNewsHeadlines(result) {
  const headlines = Array.isArray(result.news_headlines) ? result.news_headlines : [];
  const newsPanel = elements.newsPanel;

  if (!headlines.length) {
    newsPanel.innerHTML = '<p class="news-empty">No news headlines found for this topic.</p>';
    return;
  }

  const latest = headlines.filter((item) => item.category === "latest");
  const popular = headlines.filter((item) => item.category === "popular");

  newsPanel.innerHTML = "";

  const latestTitle = document.createElement("p");
  latestTitle.className = "news-section-title";
  latestTitle.textContent = "Latest Headlines";
  newsPanel.appendChild(latestTitle);

  latest.forEach((item) => newsPanel.appendChild(buildHeadlineRow(item)));

  const divider = document.createElement("hr");
  divider.className = "news-divider";
  newsPanel.appendChild(divider);

  const popularTitle = document.createElement("p");
  popularTitle.className = "news-section-title";
  popularTitle.textContent = "Most Covered";
  newsPanel.appendChild(popularTitle);

  popular.forEach((item) => newsPanel.appendChild(buildHeadlineRow(item)));
}

function renderTopPosts(result) {
  const posts = Array.isArray(result.top_posts) ? result.top_posts : [];
  elements.togglePostsButton.textContent = `Show (${posts.length}) ▼`;
  elements.topPostsContainer.classList.add("hidden");
  elements.topPostsContainer.innerHTML = "";

  posts.forEach((post) => {
    const row = document.createElement("div");
    row.className = "post-row";

    const left = document.createElement("div");
    left.className = "post-left";

    const sourceBadge = document.createElement("span");
    sourceBadge.className = `source-badge ${post.source || ""}`;
    sourceBadge.textContent = { hackernews: "HN", bluesky: "BSky", stackexchange: "SE" }[post.source] || post.source || "SRC";

    const title = document.createElement("span");
    title.className = "post-title";
    title.textContent = post.title || "Untitled";
    title.title = post.title || "Untitled";

    left.appendChild(sourceBadge);
    left.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "post-meta";
    meta.textContent = `${post.score ?? 0} pts · ${post.num_comments ?? 0} comments`;

    const link = document.createElement("a");
    link.href = post.url || "#";
    link.textContent = "↗";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    meta.appendChild(link);

    row.appendChild(left);
    row.appendChild(meta);
    elements.topPostsContainer.appendChild(row);
  });
}

function renderTopics(result) {
  elements.topicsContainer.innerHTML = "";
  const topics = Array.isArray(result.topics) ? result.topics : [];

  topics.forEach((topic) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = topic;
    chip.addEventListener("click", async () => {
      elements.queryInput.value = topic;
      await triggerSearch();
    });
    elements.topicsContainer.appendChild(chip);
  });
}

function renderResult(result) {
  currentResult = result;
  elements.resultsSection.classList.remove("hidden");

  elements.resultQuery.textContent = result.query || "";
  elements.summaryText.textContent = result.summary || "";
  elements.resultTimestamp.textContent = `Generated at ${new Date(result.generated_at).toLocaleString()}`;

  elements.summaryBadges.innerHTML = "";
  const sources = Array.isArray(result.sources_used) ? result.sources_used : [];
  sources.forEach((source) => {
    elements.summaryBadges.appendChild(createBadge(source));
  });
  elements.summaryBadges.appendChild(createBadge(`${result.post_count ?? 0} discussions`));
  elements.summaryBadges.appendChild(createBadge(result.sentiment || "neutral", result.sentiment || "neutral"));
  elements.summaryBadges.appendChild(createBadge(result.confidence || "unknown", result.confidence || ""));

  renderTopics(result);
  renderEntities(result);
  renderTopPosts(result);
  renderNewsHeadlines(result);
}

function renderHistory() {
  elements.historyList.innerHTML = "";

  if (!historyCache.length) {
    elements.historyEmpty.classList.remove("hidden");
    return;
  }

  elements.historyEmpty.classList.add("hidden");

  historyCache.forEach((item) => {
    const row = document.createElement("article");
    row.className = "history-item";

    const title = document.createElement("strong");
    title.textContent = item.query;

    const meta = document.createElement("p");
    meta.textContent = `${(item.sources_used || []).join(", ")} · ${item.post_count} posts · ${item.sentiment} · ${new Date(item.generated_at).toLocaleString()}`;

    row.appendChild(title);
    row.appendChild(meta);
    row.addEventListener("click", () => renderResult(item));

    elements.historyList.appendChild(row);
  });
}

async function fetchJson(url, options = {}) {
  fetchAbortController = new AbortController();
  const response = await fetch(url, {
    ...options,
    signal: fetchAbortController.signal
  });

  const data = await response.json();
  if (!response.ok) {
    const message = data.message || "Request failed.";
    const detail = data.detail || "No detail provided.";
    throw new Error(`${message} ${detail}`.trim());
  }

  return data;
}

async function loadHistory() {
  try {
    const data = await fetchJson(`${API_BASE}/api/history`);
    historyCache = Array.isArray(data) ? data : [];
    renderHistory();
  } catch (error) {
    showInlineError(error.message || "Failed to load history.");
  }
}

async function clearHistory() {
  try {
    await fetchJson(`${API_BASE}/api/history`, { method: "DELETE" });
    historyCache = [];
    renderHistory();
  } catch (error) {
    showInlineError(error.message || "Failed to clear history.");
  }
}

async function pollHealthUntilReady() {
  if (healthPollTimer) {
    clearInterval(healthPollTimer);
  }

  const poll = async () => {
    try {
      const data = await fetchJson(`${API_BASE}/api/health`);
      if (data.model_loaded) {
        elements.healthDot.classList.remove("dot-red");
        elements.healthDot.classList.add("dot-green");
        elements.healthText.textContent = "Ready";
        setSearchEnabled(true);
        clearInterval(healthPollTimer);
        healthPollTimer = null;
      }
    } catch (_error) {
      elements.healthDot.classList.remove("dot-green");
      elements.healthDot.classList.add("dot-red");
      elements.healthText.textContent = "Loading models...";
      setSearchEnabled(false);
    }
  };

  await poll();
  healthPollTimer = setInterval(poll, 3000);
}

function updateProgressFromPayload(payload) {
  if (payload.stage && payload.message) {
    const isDone = payload.done === true;
    markStage(Number(payload.stage), payload.message, isDone);
  }
}

function buildSearchUrl() {
  const params = new URLSearchParams();
  params.set("query", elements.queryInput.value.trim());
  params.set("limit", elements.limitSelect.value);
  params.set("time_filter", elements.timeFilter.value);
  return `${API_BASE}/api/search/stream?${params.toString()}`;
}

async function handleSsePayload(payload) {
  if (payload.error) {
    showInlineError(payload.message || "Search failed.");
    if (payload.detail) {
      showInlineError(`${payload.message} ${payload.detail}`.trim());
    }
    cancelSearch();
    return;
  }

  updateProgressFromPayload(payload);

  if (payload.done && payload.result) {
    closeEventSource();
    renderResult(payload.result);
    historyCache = [payload.result, ...historyCache].sort(
      (a, b) => new Date(b.generated_at) - new Date(a.generated_at)
    );
    renderHistory();
    resetSearchUI();
  }
}

async function startSearch() {
  const query = elements.queryInput.value.trim();
  if (query.length < 2) {
    showInlineError("Please enter at least 2 characters.");
    return;
  }

  if (requestInFlight) {
    return;
  }

  clearInlineError();
  requestInFlight = true;
  setSearchEnabled(false);
  resetStages();
  elements.progressSection.classList.remove("hidden");
  closeEventSource();

  currentEventSource = new EventSource(buildSearchUrl());

  currentEventSource.onmessage = async (event) => {
    try {
      const payload = JSON.parse(event.data);
      await handleSsePayload(payload);
    } catch (_error) {
      showInlineError("Received invalid stream data from server.");
      cancelSearch();
    }
  };

  currentEventSource.onerror = () => {
    showInlineError("Stream connection was interrupted.");
    cancelSearch();
  };
}

async function triggerSearch() {
  if (searchDebounceTimer) {
    clearTimeout(searchDebounceTimer);
  }

  await new Promise((resolve) => {
    searchDebounceTimer = setTimeout(resolve, 300);
  });

  await startSearch();
}

async function copySummary() {
  if (!currentResult || !currentResult.summary) {
    showInlineError("No summary available to copy.");
    return;
  }

  try {
    await navigator.clipboard.writeText(currentResult.summary);
  } catch (_error) {
    showInlineError("Failed to copy summary to clipboard.");
  }
}

function exportResultJson() {
  if (!currentResult) {
    showInlineError("No result available to export.");
    return;
  }

  const fileName = `${(currentResult.query || "summary").toLowerCase().replace(/[^a-z0-9_-]+/g, "_")}_${Date.now()}.json`;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

function togglePosts() {
  const isHidden = elements.topPostsContainer.classList.contains("hidden");
  if (isHidden) {
    elements.topPostsContainer.classList.remove("hidden");
    const count = (currentResult && currentResult.top_posts && currentResult.top_posts.length) || 0;
    elements.togglePostsButton.textContent = "Hide ▲";
    if (!count) {
      elements.topPostsContainer.textContent = "No source discussions available.";
    }
    return;
  }

  elements.topPostsContainer.classList.add("hidden");
  const count = (currentResult && currentResult.top_posts && currentResult.top_posts.length) || 0;
  elements.togglePostsButton.textContent = `Show (${count}) ▼`;
}

function bindEvents() {
  elements.searchButton.addEventListener("click", async () => {
    await triggerSearch();
  });

  elements.cancelButton.addEventListener("click", () => {
    cancelSearch();
  });

  elements.queryInput.addEventListener("keydown", async (event) => {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      await triggerSearch();
    }
  });

  elements.copySummaryButton.addEventListener("click", async () => {
    await copySummary();
  });

  elements.exportJsonButton.addEventListener("click", () => {
    exportResultJson();
  });

  elements.togglePostsButton.addEventListener("click", () => {
    togglePosts();
  });

  elements.clearHistoryButton.addEventListener("click", async () => {
    await clearHistory();
  });
}

async function initialize() {
  bindEvents();
  await pollHealthUntilReady();
  await loadHistory();
}

initialize();