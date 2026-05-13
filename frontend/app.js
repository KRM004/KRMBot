/**
 * app.js — Logique du chatbot RAG
 */

// ─── 1. CONFIGURATION & ÉTAT ─────────────────────────────
const CONFIG = {
  apiUrl:      localStorage.getItem('apiUrl')      || 'http://localhost:5000',
  apiEndpoint: localStorage.getItem('apiEndpoint') || '/chat',
  theme:       localStorage.getItem('theme')       || 'dark',
};

let activeFilter  = 'all';
let isLoading     = false;
let messageCount  = 0;


// ─── 2. SÉLECTION DES ÉLÉMENTS DOM ───────────────────────
const DOM = {
  chatArea:          document.getElementById('chatArea'),
  messagesContainer: document.getElementById('messagesContainer'),
  userInput:         document.getElementById('userInput'),
  sendBtn:           document.getElementById('sendBtn'),
  typingIndicator:   document.getElementById('typingIndicator'),
  welcomeScreen:     document.getElementById('welcomeScreen'),
  charCount:         document.getElementById('charCount'),
  statusDot:         document.getElementById('statusDot'),
  statusText:        document.getElementById('statusText')
};


// ─── 3. INITIALISATION & LISTENERS ───────────────────────
function init() {
  checkApiConnection();
  setupEventListeners();
}

function setupEventListeners() {
  // Filtres d'intervenants
  document.querySelectorAll('.speaker-item').forEach(item => {
    item.addEventListener('click', () => handleSpeakerFilter(item));
  });

  // Boutons de questions rapides
  document.querySelectorAll('.quick-item, .chip').forEach(item => {
    item.addEventListener('click', () => sendMessage(item.dataset.q));
  });

  // Gestion de la zone de saisie
  DOM.userInput.addEventListener('input', handleInputResize);
  DOM.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!DOM.sendBtn.disabled && !isLoading) sendMessage();
    }
  });

  // Bouton d'envoi
  DOM.sendBtn.addEventListener('click', () => {
    if (!isLoading) sendMessage();
  });
}


// ─── 4. LOGIQUE D'INTERFACE ──────────────────────────────
function handleSpeakerFilter(item) {
  document.querySelectorAll('.speaker-item').forEach(i => i.classList.remove('active'));
  item.classList.add('active');
  activeFilter = item.dataset.filter; 
  
  DOM.userInput.placeholder = activeFilter === 'all' 
    ? "Posez une question sur la réunion..." 
    : `Que voulez-vous demander sur ${activeFilter} ?`;
}

function handleInputResize() {
  const textLength = DOM.userInput.value.length;
  DOM.charCount.textContent = `${textLength}/500`;
  DOM.sendBtn.disabled = DOM.userInput.value.trim().length === 0;
  
  DOM.userInput.style.height = 'auto';
  DOM.userInput.style.height = Math.min(DOM.userInput.scrollHeight, 120) + 'px';
}

function scrollToBottom() {
  setTimeout(() => { 
    DOM.chatArea.scrollTo({ top: DOM.chatArea.scrollHeight, behavior: 'smooth' }); 
  }, 50);
}


// ─── 5. APPEL API & ENVOI DE MESSAGES ────────────────────
async function sendMessage(text) {
  const question = (text || DOM.userInput.value).trim();
  if (!question || isLoading) return;

  if (DOM.welcomeScreen) DOM.welcomeScreen.style.display = 'none';

  // Réinitialisation de l'input
  DOM.userInput.value = '';
  DOM.charCount.textContent = `0/500`;
  DOM.sendBtn.disabled = true;
  DOM.userInput.style.height = 'auto';

  // Affichage utilisateur
  appendMessage('user', question);
  isLoading = true;
  DOM.typingIndicator.classList.add('show');
  scrollToBottom();

  // Appel au backend
  try {
    const result = await callApi(question);
    DOM.typingIndicator.classList.remove('show');
    
    if (result.error) {
       appendMessage('bot', `❌ Erreur du serveur : ${result.error}`);
    } else {
       appendMessage('bot', result.answer, result.sources || []);
    }
  } catch (err) {
    DOM.typingIndicator.classList.remove('show');
    if (err.message === "Failed to fetch" || err.message.includes("NetworkError")) {
        appendMessage('bot', "⚠️ Impossible de se connecter au serveur Python. Vérifiez que `python app.py` est bien lancé dans le terminal et que le port 5000 est ouvert.");
    } else {
        appendMessage('bot', `❌ Erreur : ${err.message}`);
    }
  }

  isLoading = false;
  scrollToBottom();
}

async function callApi(question) {
  const url = CONFIG.apiUrl + CONFIG.apiEndpoint;
  const body = {
    question: question,
    filter_speaker: activeFilter 
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(20000)
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
     if (data && data.error) throw new Error(data.error);
     throw new Error(`Erreur HTTP ${response.status}`);
  }
  
  return data;
}

async function checkApiConnection() {
  try {
    const res = await fetch(CONFIG.apiUrl + '/health', { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      DOM.statusDot.className = 'status-dot connected';
      DOM.statusText.textContent = 'API connectée';
    } else {
      throw new Error();
    }
  } catch {
    DOM.statusDot.className = 'status-dot error';
    DOM.statusText.textContent = 'API hors ligne';
  }
}


// ─── 6. FORMATAGE & CRÉATION DOM ─────────────────────────
function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[tag]));
}

function formatText(text) {
  return escapeHTML(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br/>');
}

function appendMessage(role, text, sources = []) {
  messageCount++;
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = formatText(text);
  msg.appendChild(bubble);

  if (role === 'bot' && sources.length > 0) {
    msg.appendChild(buildSourcesBlock(sources));
  }

  DOM.messagesContainer.appendChild(msg);
}

function buildSourcesBlock(sources) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg-sources';
  
  const toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.textContent = `▸ ${sources.length} source(s) utilisée(s)`;

  const body = document.createElement('div');
  body.className = 'sources-body';

  sources.forEach(src => {
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-header">
        <span class="source-speaker">${escapeHTML(src.speaker || 'Intervenant')}</span>
        <span class="source-dept">${escapeHTML(src.department || 'Réunion')}</span>
      </div>
      <p class="source-text">"${escapeHTML(src.content || 'Extrait non disponible...')}"</p>
    `;
    body.appendChild(card);
  });

  toggle.addEventListener('click', () => {
    body.classList.toggle('open');
    toggle.textContent = body.classList.contains('open') 
      ? `▾ ${sources.length} source(s) utilisée(s)` 
      : `▸ ${sources.length} source(s) utilisée(s)`;
  });

  wrapper.appendChild(toggle);
  wrapper.appendChild(body);
  return wrapper;
}

// ─── DÉMARRAGE ───────────────────────────────────────────
init();