const $ = (id) => document.getElementById(id);
const text = $('text');
let voices = [];

function progress(title, percent, detail) {
  $('profileProgress').hidden = false;
  $('progressTitle').textContent = title;
  $('progressPercent').textContent = `${percent}%`;
  $('progressBar').style.width = `${percent}%`;
  $('progressDetail').textContent = detail;
}

function updateCharacters() {
  $('characters').textContent = `${text.value.length.toLocaleString('it-IT')} caratteri`;
}

function bytes(value) {
  return value < 1024 * 1024 ? `${Math.ceil(value / 1024)} KB` : `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function renderVoices(selectedId = $('voiceSelect').value) {
  const list = $('voiceList');
  const select = $('voiceSelect');
  select.innerHTML = '<option value="">Scegli una voce salvata…</option>';
  if (!voices.length) {
    list.innerHTML = '<p class="muted">Non hai ancora salvato una voce.</p>';
  } else {
    list.innerHTML = voices.map((voice) => `<div class="voice-row"><div><b>${escapeHtml(voice.name)}</b><small>Profilo locale · ${bytes(voice.size)}${voice.prepared ? ' · campione ottimizzato' : ' · verrà ottimizzato alla prima generazione'}</small></div><button class="delete-voice" data-id="${voice.id}" type="button">Elimina</button></div>`).join('');
    list.querySelectorAll('.delete-voice').forEach((button) => button.addEventListener('click', () => deleteVoice(button.dataset.id)));
  }
  voices.forEach((voice) => {
    const option = document.createElement('option');
    option.value = voice.id;
    option.textContent = voice.name;
    option.selected = voice.id === selectedId;
    select.append(option);
  });
  $('newVoicePanel').hidden = voices.length >= 3;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}

async function loadVoices(selectedId) {
  const response = await fetch('/api/voices');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Impossibile leggere le voci salvate.');
  voices = data.voices;
  renderVoices(selectedId);
}

async function deleteVoice(id) {
  const voice = voices.find((item) => item.id === id);
  if (!voice || !confirm(`Eliminare definitivamente “${voice.name}” dal PC?`)) return;
  const response = await fetch(`/api/voices/${id}`, { method: 'DELETE' });
  if (!response.ok) return alert('Non riesco a eliminare questa voce.');
  await loadVoices();
}

text.addEventListener('input', updateCharacters);
updateCharacters();

$('saveVoiceButton').addEventListener('click', async () => {
  const name = $('voiceName').value.trim();
  const file = $('reference').files[0];
  if (!name) return progress('Serve un nome', 0, 'Esempio: Enzo Ufficiale.');
  if (!file) return progress('Manca il campione', 0, 'Scegli un file audio della voce da salvare.');
  progress('Salvataggio della voce', 35, 'Il campione viene copiato solo nella cartella locale dell’app.');
  try {
    const response = await fetch('/api/voices', {
      method: 'POST', headers: { 'content-type': file.type, 'x-voice-name': encodeURIComponent(name) }, body: file
    });
    const data = await response.json();
    if (!response.ok) return progress('Impossibile salvare', 0, data.error || 'Errore inatteso.');
    await loadVoices(data.voice.id);
    $('voiceName').value = '';
    $('reference').value = '';
    progress('Voce salvata', 100, `“${data.voice.name}” è disponibile nel menu “Voce da usare”.`);
  } catch {
    progress('Errore di connessione', 0, 'Controlla che START.bat sia attivo, poi riprova.');
  }
});

const stepTitles = {
  searching: 'Ricerca GPU', creating: 'Noleggio GPU', booting: 'Avvio GPU', loading: 'Caricamento modello',
  uploading: 'Invio dati', generating: 'Generazione', downloading: 'Download', destroying: 'Distruzione GPU',
  done: 'Completato', error: 'Errore'
};

function jobProgress(job) {
  $('jobProgress').hidden = false;
  $('jobTitle').textContent = stepTitles[job.status] || job.status;
  $('jobPercent').textContent = `${Math.round(job.percent)}%`;
  $('jobBar').style.width = `${job.percent}%`;
  const cost = job.rate ? ` · ${job.gpu} · spesa finora $${job.cost.toFixed(3)}` : '';
  $('jobDetail').textContent = (job.status === 'error' ? job.error : job.detail) + cost;
}

function setBusy(busy) {
  $('generateButton').disabled = busy;
  $('cancelButton').hidden = !busy;
}

async function followJob(id) {
  for (;;) {
    const response = await fetch(`/api/jobs/${id}`);
    const { job, error } = await response.json();
    if (!response.ok) throw new Error(error);
    jobProgress(job);
    if (job.status === 'done') return job;
    if (job.status === 'error') {
      // The server destroys the GPU before reporting the error; say so explicitly.
      $('jobDetail').textContent = `${job.error}${job.instanceId ? (job.destroyed ? ' · GPU distrutta.' : ' · ATTENZIONE: distruzione non confermata, controlla cloud.vast.ai.') : ''}`;
      return null;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

let currentJob = null;
$('cancelButton').addEventListener('click', async () => {
  if (!currentJob || !confirm('Annullare la generazione? La GPU verrà distrutta.')) return;
  $('cancelButton').disabled = true;
  await fetch(`/api/jobs/${currentJob}/cancel`, { method: 'POST' });
  $('cancelButton').disabled = false;
});

$('generateButton').addEventListener('click', async () => {
  const voiceId = $('voiceSelect').value;
  $('result').hidden = true;
  $('audioOutput').hidden = true;
  const payload = {
    voiceId, text: text.value, language: $('language').value, characters: text.value.length,
    hourlyRate: Number($('rate').value), minutes: Number($('minutes').value), safetyMinutes: 10,
    budgetCap: Number($('budget').value), confirmDestroy: $('destroy').checked, pause: Number($('pause').value),
    keepAlive: document.querySelector('input[name="lifecycle"]:checked').value === 'keep',
    stability: $('stability').value, quality: $('quality').value, speed: Number($('speed').value)
  };
  try {
    const response = await fetch('/api/jobs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) {
      $('result').hidden = false;
      $('result').innerHTML = `<h2>Controlla il job</h2><p>${escapeHtml(data.error)}</p>`;
      return;
    }
    currentJob = data.id;
    setBusy(true);
    const job = await followJob(data.id);
    if (job) {
      $('player').src = job.files.full;
      $('downloadAudio').href = `${job.files.full}?download`;
      $('downloadParts').href = `${job.files.parts}?download`;
      const check = job.quality ? ` · fedeltà al testo ${Math.round(job.quality * 100)}%${job.weakest?.length ? ` (da riascoltare: blocchi ${job.weakest.join(', ')})` : ''}` : '';
      $('outputInfo').textContent = `${job.chunks} blocchi · ${job.audioSeconds} secondi di audio · costo GPU stimato $${job.cost}${check}`;
      $('audioOutput').hidden = false;
      await loadVoices(voiceId);
      loadHistory();
    }
  } catch {
    $('result').hidden = false;
    $('result').innerHTML = '<h2>Server non raggiungibile</h2><p>Chiudi e riapri START.bat, poi ricarica questa pagina. Al riavvio l’app distrugge eventuali GPU rimaste attive.</p>';
  } finally {
    currentJob = null;
    setBusy(false);
  }
});

// History of finished generations, read from outputs/ on the server.
const stabilityLabels = { natural: 'naturale', stable: 'stabile', very: 'molto stabile' };

function duration(seconds) {
  const total = Math.round(seconds || 0);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

function renderHistory(entries) {
  const list = $('historyList');
  if (!entries.length) {
    list.innerHTML = '<p class="muted">Ancora nessuna generazione.</p>';
    return;
  }
  list.innerHTML = entries.map((entry) => {
    const base = `/api/jobs/${entry.id}/files`;
    const when = new Date(entry.createdAt).toLocaleString('it-IT', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
    const preview = entry.text
      ? `<p class="history-text">${escapeHtml(entry.text.length > 220 ? `${entry.text.slice(0, 220)}…` : entry.text)}</p>`
      : '<p class="history-text missing">Testo non salvato (generazione precedente alla cronologia).</p>';
    const meta = [
      duration(entry.audioSeconds),
      entry.chunks ? `${entry.chunks} blocchi` : null,
      entry.stability ? `lettura ${stabilityLabels[entry.stability] || entry.stability}` : null,
      entry.speed && entry.speed !== 1 ? `velocità ${Math.round(entry.speed * 100)}%` : null,
      entry.takes > 1 ? `qualità alta${entry.quality ? ` (fedeltà ${Math.round(entry.quality * 100)}%)` : ''}` : null,
      entry.cost != null ? `$${entry.cost}` : null,
      bytes(entry.bytes)
    ].filter(Boolean).join(' · ');
    return `<article class="history-item">
      <div class="history-head"><b>${escapeHtml(entry.voiceName || 'Generazione')}</b><span>${when}</span></div>
      ${preview}
      <div class="history-meta">${meta}</div>
      <audio controls preload="none" src="${base}/clova-voice.wav"></audio>
      <div class="actions">
        <a class="download" href="${base}/clova-voice.wav?download">File unico (WAV)</a>
        ${entry.hasParts ? `<a class="download" href="${base}/clova-voice-blocchi.zip?download">Più file (ZIP)</a>` : ''}
        ${entry.text ? `<button type="button" class="ghost" data-reuse="${entry.id}">Riusa il testo</button>` : ''}
        <button type="button" class="secondary" data-delete="${entry.id}">Elimina</button>
      </div>
    </article>`;
  }).join('');
  list.querySelectorAll('[data-reuse]').forEach((button) => button.addEventListener('click', () => {
    text.value = entries.find((entry) => entry.id === button.dataset.reuse).text;
    updateCharacters();
    text.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));
  list.querySelectorAll('[data-delete]').forEach((button) => button.addEventListener('click', async () => {
    if (!confirm('Eliminare definitivamente questa generazione dal PC?')) return;
    const response = await fetch(`/api/history/${button.dataset.delete}`, { method: 'DELETE' });
    if (!response.ok) alert((await response.json()).error || 'Non riesco a eliminarla.');
    loadHistory();
  }));
}

async function loadHistory() {
  try {
    const response = await fetch('/api/history');
    if (response.ok) renderHistory((await response.json()).entries);
  } catch { /* Server closed: keep the current list. */ }
}

loadHistory();

// Pronunciation dictionary, saved in data/pronunce.txt and applied on the server.
async function loadPronunciations() {
  const response = await fetch('/api/pronunciations');
  if (response.ok) $('pronunciations').value = (await response.json()).text;
}

$('savePronunciations').addEventListener('click', async () => {
  $('pronunciationStatus').textContent = 'Salvataggio…';
  try {
    const response = await fetch('/api/pronunciations', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: $('pronunciations').value }) });
    const data = await response.json();
    $('pronunciationStatus').textContent = response.ok ? `Salvato: ${data.entries} parole.` : data.error;
  } catch {
    $('pronunciationStatus').textContent = 'Server non raggiungibile.';
  }
});

loadPronunciations().catch(() => {});

// Kept-alive GPU: always-visible banner with countdown, cost and a destroy button.
let instance = { active: false };

function renderInstance() {
  $('instanceBanner').hidden = !instance.active;
  if (!instance.active) return;
  const parts = [instance.gpu, `spesa $${instance.cost.toFixed(3)}`];
  if (instance.busy || !instance.keepUntil) parts.push('in uso');
  else {
    const left = Math.max(0, Math.round((instance.keepUntil - Date.now()) / 1000));
    parts.push(`si distrugge da sola tra ${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}`);
  }
  $('instanceInfo').textContent = `· ${parts.join(' · ')}`;
}

async function pollInstance() {
  try {
    const response = await fetch('/api/instance');
    if (response.ok) instance = await response.json();
  } catch { /* Server closed: keep the last state. */ }
  renderInstance();
}

$('destroyInstanceButton').addEventListener('click', async () => {
  const message = instance.busy ? 'C’è una generazione in corso: verrà interrotta. Distruggere la GPU?' : 'Distruggere subito la GPU?';
  if (!confirm(message)) return;
  $('destroyInstanceButton').disabled = true;
  $('destroyInstanceButton').textContent = 'Distruzione in corso…';
  try {
    const response = await fetch('/api/instance/destroy', { method: 'POST' });
    const data = await response.json();
    if (!data.destroyed) alert('Non riesco a confermare la distruzione: controlla su cloud.vast.ai.');
  } finally {
    $('destroyInstanceButton').disabled = false;
    $('destroyInstanceButton').textContent = 'Distruggi istanza';
    pollInstance();
  }
});

window.addEventListener('beforeunload', (event) => {
  if (!instance.active) return;
  event.preventDefault();
  event.returnValue = 'Attenzione: la GPU è ancora attiva.';
});

pollInstance();
setInterval(pollInstance, 3000);
setInterval(renderInstance, 1000);

loadVoices().catch((error) => { $('voiceList').innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`; });
