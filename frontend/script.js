// API Configuration
const API_BASE_URL = '/api';
let userPersona = 'explorer'; // Default

// Session Management
let sessionId = localStorage.getItem('nyaya_session_id');
if (!sessionId) {
    sessionId = 'session_' + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('nyaya_session_id', sessionId);
}

const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const onboardingOverlay = document.getElementById('onboarding-overlay');
const appContainer = document.querySelector('.app-container');
const switchBtn = document.getElementById('switch-btn');
const profileStatus = document.getElementById('profile-status');
const attachBtn = document.getElementById('attach-btn');
const fileUpload = document.getElementById('file-upload');
const personaTooltip = document.getElementById('persona-tooltip');
const infoTrigger = document.getElementById('info-trigger');
let isRequestInProgress = false;
let latestMemoText = "";

const newSessionBtn = document.getElementById('new-session-btn');
if (newSessionBtn) {
    newSessionBtn.addEventListener('click', () => {
        if (confirm("Start a new session? This will clear current context.")) {
            localStorage.removeItem('nyaya_session_id');
            window.location.reload();
        }
    });
}

const viewMemoBtn = document.getElementById('view-memo-btn');
if (viewMemoBtn) {
    viewMemoBtn.addEventListener('click', () => {
        if (latestMemoText) {
            exportToPdf(latestMemoText);
        } else {
            alert("No memo available yet.");
        }
    });
}

const evalRagBtn = document.getElementById('eval-rag-btn');
if (evalRagBtn) {
    evalRagBtn.addEventListener('click', async () => {
        if (isRequestInProgress) return;
        setLockdown(true);
        addMessage("Initiating RAG Pipeline Evaluation (this may take a minute)...", 'user');
        
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.classList.add('message', 'system', 'glass', 'legal-loader');
        loadingDiv.innerHTML = `
            <div class="gavel-animate"><i data-lucide="gavel"></i></div>
            <div class="loader-text">Evaluating Ragas Metrics...</div>
        `;
        chatMessages.appendChild(loadingDiv);
        lucide.createIcons();
        scrollToBottom();

        try {
            const response = await fetch(`${API_BASE_URL}/evaluate_rag`, { method: 'POST' });
            const data = await response.json();
            document.getElementById(loadingId).remove();
            
            if (data.rating !== undefined) {
                addMessage(`RAG Pipeline Evaluation Complete!\n\n**Overall Rating:** ${data.rating} / 5.0`, 'system');
            } else if (data.detail) {
                addMessage(`Evaluation Error: ${data.detail}`, 'system');
            }
        } catch (error) {
            if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
            addMessage(`Connection Error during evaluation.`, 'system');
        } finally {
            setLockdown(false);
        }
    });
}

// Persona Capability Data
const personaCapabilities = {
    lawyer: "<strong>Legal Professional Mode</strong><br>Optimized for deep statutory analysis, procedural advice, and drafting formal legal documents with precise terminology.",
    student: "<strong>Academic Mode</strong><br>Focuses on legal theory, landmark Supreme Court cases, and historical context of Indian laws.",
    citizen: "<strong>Empowerment Mode</strong><br>Provides simple explanations of legal rights, RTI support, and step-by-step guidance for basic legal issues.",
    explorer: "<strong>Orientation Mode</strong><br>General overview of the Indian judicial hierarchy, history, and fundamental legal concepts."
};

function setLockdown(active) {
    isRequestInProgress = active;
    if (active) {
        document.body.classList.add('lockdown-active');
        userInput.placeholder = "Judicial Lockdown: Analysis in progress...";
        sendBtn.disabled = true;
    } else {
        document.body.classList.remove('lockdown-active');
        userInput.placeholder = "Enter your legal query here...";
        sendBtn.disabled = false;
    }
}

// Handle File Upload
attachBtn.addEventListener('click', () => fileUpload.click());

fileUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    addMessage(`Uploading ${file.name}...`, 'user');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.text) {
            userInput.value = `Audit this document: \n\n ${data.text.substring(0, 1000)}...`;
            addMessage(`Document ${file.name} processed. I have extracted the text. Click Execute to start the audit.`, 'system');
        }
    } catch (err) {
        addMessage(`Error uploading file: ${err.message}`, 'system');
    }
});
document.querySelectorAll('.persona-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        userPersona = btn.getAttribute('data-persona');
        onboardingOverlay.classList.add('hidden');
        appContainer.classList.remove('blur-background');
        
        // Update top bar status
        profileStatus.innerText = userPersona.charAt(0).toUpperCase() + userPersona.slice(1) + " Dashboard";

        // Apply Dynamic Theme
        document.documentElement.setAttribute('data-theme', userPersona);
        
        // Update Wisdom Tooltip
        personaTooltip.innerHTML = personaCapabilities[userPersona];
        
        addMessage(`Welcome! You are now connected as a **${userPersona.toUpperCase()}**. How can I help you navigate the legal system today?`, 'system');
    });
});

function highlightCitations(text) {
    // Regex to find common Indian legal citations (e.g., Section 302, Article 21, IPC 302)
    const citationRegex = /\b(Section|Article|Act|Clause)\s+\d+[A-Z]?\b|\b(IPC|BNS|CRPC|CPC|DPDPA)\s+\d+[A-Z]?\b/gi;
    return text.replace(citationRegex, (match) => {
        return `<span class="legal-citation" title="Judicial Reference">${match}</span>`;
    });
}

switchBtn.addEventListener('click', () => {
    onboardingOverlay.classList.remove('hidden');
    appContainer.classList.add('blur-background');
});

// Motto Carousel Data
const mottos = [
    { text: "सत्यमेव जयते", lang: "Hindi" },
    { text: "সত্যমেব জয়তে", lang: "Bengali" },
    { text: "सत्यमेव जयते", lang: "Marathi" },
    { text: "సత్యమేవ జయతే", lang: "Telugu" },
    { text: "சத்யமேவ ஜெயதே", lang: "Tamil" },
    { text: "સત્યમેવ જયતે", lang: "Gujarati" },
    { text: "ستیہ میو جیتے", lang: "Urdu" },
    { text: "ಸತ್ಯಮೇವ ಜಯತೇ", lang: "Kannada" },
    { text: "ସତ୍ୟମେବ ଜୟତେ", lang: "Odia" },
    { text: "സത്യമേവ ജയതേ", lang: "Malayalam" }
];

let currentMottoIndex = 0;
const mottoText = document.getElementById('motto-text');
const mottoLang = document.getElementById('motto-lang');

function updateMotto() {
    mottoText.classList.add('fade-out');
    mottoLang.classList.add('fade-out');

    setTimeout(() => {
        currentMottoIndex = (currentMottoIndex + 1) % mottos.length;
        mottoText.innerText = mottos[currentMottoIndex].text;
        mottoLang.innerText = mottos[currentMottoIndex].lang;
        
        mottoText.classList.remove('fade-out');
        mottoLang.classList.remove('fade-out');
    }, 500);
}

setInterval(updateMotto, 4000); // Change every 4 seconds

function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`; // Explicitly set classes
    
    // Highlight Citations
    let highlightedText = highlightCitations(text);
    
    // Add Spark Icon for System
    if (sender === 'system') {
        const spark = document.createElement('div');
        spark.className = 'system-spark';
        spark.innerHTML = '✨';
        msgDiv.appendChild(spark);
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = marked.parse(highlightedText);
    msgDiv.appendChild(contentDiv);

    // Action Button Detection (Proactive Agent Feature)
    if (sender === 'system') {
        const actionArea = document.createElement('div');
        actionArea.classList.add('action-area');

        if (text.toLowerCase().includes('rti query') || text.toLowerCase().includes('rti act')) {
            const btn = createActionButton('scroll', 'Draft RTI Query', () => {
                userInput.value = "Draft an RTI query for this matter.";
                handleSendMessage();
            });
            actionArea.appendChild(btn);
        }

        if (text.toLowerCase().includes('legal notice')) {
            const btn = createActionButton('file-text', 'Prepare Legal Notice', () => {
                userInput.value = "Help me prepare a formal legal notice for this.";
                handleSendMessage();
            });
            actionArea.appendChild(btn);
        }

        // New: Copy Draft Detection
        if (text.includes('LEGAL NOTICE') || text.includes('Subject: Application under RTI') || text.includes('Facts:')) {
            const copyBtn = createActionButton('copy', 'Copy Draft', () => {
                navigator.clipboard.writeText(text);
                alert('Draft copied to clipboard! (Satyameva Jayate)');
            });
            actionArea.appendChild(copyBtn);

            const docBtn = createActionButton('file-text', 'Download DOC', () => exportToDoc(text));
            actionArea.appendChild(docBtn);

            const pdfBtn = createActionButton('download', 'Download PDF', () => exportToPdf(text));
            actionArea.appendChild(pdfBtn);
        }

        if (actionArea.children.length > 0) {
            msgDiv.appendChild(actionArea);
        }
    }
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const timeSpan = document.createElement('span');
    timeSpan.classList.add('message-time');
    timeSpan.innerText = time;
    msgDiv.appendChild(timeSpan);
    
    chatMessages.appendChild(msgDiv);
    setTimeout(scrollToBottom, 50); // Small delay for smooth rendering
    lucide.createIcons();
}

function scrollToBottom() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

function exportToDoc(text) {
    const header = "<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'><head><meta charset='utf-8'><title>Legal Draft</title></head><body>";
    const footer = "</body></html>";
    const sourceHTML = header + marked.parse(text) + footer;
    
    const source = 'data:application/vnd.ms-word;charset=utf-8,' + encodeURIComponent(sourceHTML);
    const fileDownload = document.createElement("a");
    document.body.appendChild(fileDownload);
    fileDownload.href = source;
    fileDownload.download = 'Nyaya_Legal_Draft.doc';
    fileDownload.click();
    document.body.removeChild(fileDownload);
}

function exportToPdf(text) {
    // Create an in-page print modal (avoids popup blockers in HF iframe)
    const existingModal = document.getElementById('nyaya-print-modal');
    if (existingModal) existingModal.remove();

    const modal = document.createElement('div');
    modal.id = 'nyaya-print-modal';
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: white; z-index: 99999; overflow-y: auto;
        font-family: 'Times New Roman', serif; padding: 50px;
        line-height: 1.6; color: #000; box-sizing: border-box;
    `;

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕ Close';
    closeBtn.style.cssText = `
        position: fixed; top: 20px; right: 20px;
        padding: 8px 16px; background: #4a0e0e; color: white;
        border: none; border-radius: 6px; cursor: pointer; font-size: 14px;
        z-index: 100000;
    `;
    closeBtn.onclick = () => modal.remove();

    const printBtn = document.createElement('button');
    printBtn.textContent = '🖨 Print / Save as PDF';
    printBtn.style.cssText = `
        position: fixed; top: 20px; right: 160px;
        padding: 8px 16px; background: #c5a059; color: white;
        border: none; border-radius: 6px; cursor: pointer; font-size: 14px;
        z-index: 100000;
    `;
    printBtn.onclick = () => window.print();

    const content = document.createElement('div');
    content.id = 'nyaya-print-content';
    content.innerHTML = `
        <div style="text-align:center; margin-bottom: 30px; border-bottom: 2px solid #c5a059; padding-bottom: 20px;">
            <h1 style="color: #4a0e0e; margin: 0;">⚖️ Nyaya Agent — Legal Draft</h1>
            <p style="color: #666; margin: 5px 0;">Generated by Nyaya AI Legal Assistant</p>
        </div>
        <div style="max-width: 800px; margin: 0 auto;">
            ${marked.parse(text)}
        </div>
    `;

    modal.appendChild(closeBtn);
    modal.appendChild(printBtn);
    modal.appendChild(content);
    document.body.appendChild(modal);

    // Add print CSS so only the memo content prints cleanly
    const printStyle = document.getElementById('nyaya-print-style') || document.createElement('style');
    printStyle.id = 'nyaya-print-style';
    printStyle.textContent = `
        @media print {
            body > *:not(#nyaya-print-modal) { display: none !important; }
            #nyaya-print-modal { position: static !important; padding: 0 !important; }
            #nyaya-print-modal button { display: none !important; }
        }
    `;
    document.head.appendChild(printStyle);
}

function createActionButton(icon, label, callback) {
    const btn = document.createElement('button');
    btn.classList.add('action-btn');
    btn.innerHTML = `<i data-lucide="${icon}"></i> <span>${label}</span>`;
    btn.onclick = callback;
    return btn;
}

async function handleSendMessage() {
    if (isRequestInProgress) return;
    const text = userInput.value.trim();
    if (!text) return;

    setLockdown(true);

    // Add user message to UI
    addMessage(text, 'user');
    userInput.value = '';

    // Show animated loading indicator
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.classList.add('message', 'system', 'glass', 'legal-loader');
    loadingDiv.innerHTML = `
        <div class="gavel-animate"><i data-lucide="gavel"></i></div>
        <div class="loader-text">Order in the Court... Analyzing</div>
    `;
    chatMessages.appendChild(loadingDiv);
    lucide.createIcons();
    scrollToBottom();

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                message: text,
                persona: userPersona,
                session_id: sessionId
            })
        });

        const data = await response.json();
        
        // Remove loading indicator
        document.getElementById(loadingId).remove();

        if (data.summary) {
            const summaryText = document.getElementById('context-summary-text');
            if (summaryText) {
                summaryText.innerHTML = `<p>${data.summary}</p>`;
            }
        }
        
        // Refresh recent sessions since the DB was just updated
        fetchRecentSessions();

        if (data.response) {
            addMessage(data.response, 'system');
            
            // If the backend generated a memo, we can offer an explicit download button
            if (data.memo) {
                const memoText = data.memo.detailed_report || JSON.stringify(data.memo, null, 2);
                latestMemoText = memoText;
                const memoSection = document.getElementById('memo-actions-section');
                if (memoSection) memoSection.style.display = 'block';
                
                setTimeout(() => {
                    const msgDiv = chatMessages.lastElementChild;
                    const actionArea = msgDiv.querySelector('.action-area') || document.createElement('div');
                    actionArea.classList.add('action-area');
                    
                    const pdfBtn = createActionButton('download', 'Download Memo PDF', () => exportToPdf(memoText));
                    actionArea.appendChild(pdfBtn);
                    
                    if (!msgDiv.querySelector('.action-area')) {
                        msgDiv.appendChild(actionArea);
                    }
                }, 100);
            }
        } else if (data.detail) {
            addMessage(`Error: ${data.detail}`, 'system');
        }
    } catch (error) {
        if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
        addMessage(`Connection Error: Could not reach the Judicial Network. Ensure the server is running.`, 'system');
        console.error('API Error:', error);
    } finally {
        setLockdown(false);
    }
}

sendBtn.addEventListener('click', handleSendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSendMessage();
    }
});

addMessage("Greetings. I am **Nyaya Agent**. I am ready to assist with your legal research, drafting, or compliance queries.", "system");

// Sidebar Navigation
const toolExamples = {
    search: "Search for landmark cases. e.g., 'Find Supreme Court judgments on Right to Privacy (Justice K.S. Puttaswamy case)'",
    notice: "Prepare a formal notice. e.g., 'Help me draft a legal notice for non-payment of rent by a tenant.'",
    rti: "Draft an RTI application. e.g., 'Draft an RTI to find out the status of road repairs in Ward 12.'",
    compliance: "Audit documents for DPDPA 2023. e.g., 'Audit this privacy policy for compliance with the new data protection act.'",
    bridge: "Map old laws to new. e.g., 'What is the new BNS section for IPC 302 (Murder)?'"
};

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        if (isRequestInProgress) {
            alert('Judicial Lockdown Active: Please wait for the current analysis to complete.');
            return;
        }

        document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
        item.classList.add('active');
        
        const tool = item.getAttribute('data-tool');
        if (tool !== 'chat') {
            const example = toolExamples[tool];
            addMessage(`Switched to **${tool.toUpperCase()}** module.\n\n**Example:** ${example}`, 'system');
        }
    });
});

// History & Sessions Management
async function fetchRecentSessions() {
    try {
        const response = await fetch(`${API_BASE_URL}/sessions`);
        const data = await response.json();
        const list = document.getElementById('recent-sessions-list');
        if (list && data.sessions) {
            list.innerHTML = '';
            data.sessions.forEach(session => {
                const li = document.createElement('li');
                li.className = 'recent-item';
                li.style.display = 'flex';
                li.style.justifyContent = 'space-between';
                li.style.alignItems = 'center';
                
                const titleSpan = document.createElement('span');
                titleSpan.style.display = 'flex';
                titleSpan.style.alignItems = 'center';
                titleSpan.style.gap = '0.5rem';
                titleSpan.style.overflow = 'hidden';
                titleSpan.style.flex = '1';
                titleSpan.innerHTML = `<i data-lucide="message-circle" style="flex-shrink: 0;"></i> <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; flex: 1;">${session.title}</span>`;
                
                const deleteBtn = document.createElement('button');
                deleteBtn.innerHTML = `<i data-lucide="trash-2" style="width: 14px; height: 14px;"></i>`;
                deleteBtn.style.background = 'none';
                deleteBtn.style.border = 'none';
                deleteBtn.style.color = 'var(--text-color)';
                deleteBtn.style.opacity = '0.5';
                deleteBtn.style.cursor = 'pointer';
                deleteBtn.style.padding = '0.2rem';
                
                deleteBtn.onmouseover = () => deleteBtn.style.color = '#ff4444';
                deleteBtn.onmouseout = () => deleteBtn.style.color = 'var(--text-color)';
                
                deleteBtn.onclick = async (e) => {
                    e.stopPropagation();
                    if(confirm('Delete this chat?')) {
                        try {
                            const res = await fetch(`${API_BASE_URL}/session/${session.session_id}`, { method: 'DELETE' });
                            if (res.ok) {
                                if (session.session_id === sessionId) {
                                    document.getElementById('new-session-btn').click();
                                } else {
                                    fetchRecentSessions();
                                }
                            }
                        } catch(err) { console.error('Delete failed', err); }
                    }
                };
                
                li.appendChild(titleSpan);
                li.appendChild(deleteBtn);
                li.onclick = () => loadSession(session.session_id);
                list.appendChild(li);
            });
            lucide.createIcons();
        }
    } catch (e) {
        console.error("Failed to fetch sessions", e);
    }
}

async function loadSession(sid) {
    if (isRequestInProgress) return;
    
    // Switch session
    sessionId = sid;
    localStorage.setItem('nyaya_session_id', sid);
    
    // Clear chat
    chatMessages.innerHTML = '';
    latestMemoText = "";
    const memoSection = document.getElementById('memo-actions-section');
    if (memoSection) memoSection.style.display = 'none';
    
    const summaryText = document.getElementById('context-summary-text');
    if (summaryText) summaryText.innerHTML = '<p class="empty-state">Start chatting to build legal context...</p>';

    // Show Loader
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.classList.add('message', 'system', 'glass', 'legal-loader');
    loadingDiv.innerHTML = `
        <div class="gavel-animate"><i data-lucide="folder-open"></i></div>
        <div class="loader-text">Loading Records...</div>
    `;
    chatMessages.appendChild(loadingDiv);
    lucide.createIcons();
    setLockdown(true);

    // Fetch history
    try {
        const response = await fetch(`${API_BASE_URL}/session/${sid}`);
        const data = await response.json();
        
        document.getElementById(loadingId).remove();
        
        if (data.summary && summaryText && data.summary.trim() !== '') {
            summaryText.innerHTML = `<p>${data.summary}</p>`;
        }
        
        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(msg => {
                addMessage(msg.content, msg.role === 'assistant' ? 'system' : 'user');
            });
        } else {
            addMessage("Resumed empty session.", "system");
        }
    } catch (e) {
        console.error("Failed to load session", e);
        if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
        addMessage("Failed to retrieve court records.", "system");
    } finally {
        setLockdown(false);
    }
}

const recentsToggle = document.getElementById('recents-toggle');
const recentsList = document.getElementById('recent-sessions-list');
const recentsChevron = document.getElementById('recents-chevron');

if (recentsToggle && recentsList && recentsChevron) {
    recentsToggle.addEventListener('click', () => {
        if (recentsList.style.display === 'none') {
            recentsList.style.display = 'flex';
            recentsChevron.style.transform = 'rotate(0deg)';
        } else {
            recentsList.style.display = 'none';
            recentsChevron.style.transform = 'rotate(-90deg)';
        }
    });
}

// Fetch recent sessions on script load
fetchRecentSessions();
