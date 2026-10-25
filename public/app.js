const $ = (id) => document.getElementById(id);
const text = $('text');
let voices = [];
let voiceLimit = 5;

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
    list.innerHTML = voices.map((voice) => `<div class="voice-row${voice.type === 'trained' ? ' trained' : ''}"><div><b>${escapeHtml(voice.name)}</b><small>${voice.type === 'trained' ? 'Modello addestrato sulla tua voce · salvato su Hugging Face (privato)' : `Voce clonata · ${bytes(voice.size)}${voice.prepared ? ' · campione ottimizzato' : ' · verrà ottimizzato alla prima generazione'}`}</small></div><div class="actions">${voice.prepared ? `<button class="ghost transcript-voice" data-id="${voice.id}" type="button">Trascrizione</button>` : ''}<button class="delete-voice" data-id="${voice.id}" type="button">Elimina</button></div></div>
      <div class="transcript-panel" id="transcript-${voice.id}" hidden>
        <p class="helper">Questo è il testo che Qwen abbina al tuo campione. Deve corrispondere <b>parola per parola</b> a ciò che dici nella registrazione: correggi eventuali errori di Whisper.</p>
        <textarea rows="7"></textarea>
        <div class="actions"><button class="ghost save-transcript" data-id="${voice.id}" type="button">Salva trascrizione</button><span class="muted"></span></div>
      </div>`).join('');
    list.querySelectorAll('.delete-voice').forEach((button) => button.addEventListener('click', () => deleteVoice(button.dataset.id)));
    list.querySelectorAll('.transcript-voice').forEach((button) => button.addEventListener('click', () => toggleTranscript(button.dataset.id)));
    list.querySelectorAll('.save-transcript').forEach((button) => button.addEventListener('click', () => saveTranscript(button.dataset.id)));
  }
  voices.forEach((voice) => {
    const option = document.createElement('option');
    option.value = voice.id;
    option.textContent = voice.name;
    option.selected = voice.id === selectedId;
    select.append(option);
  });
  $('newVoicePanel').hidden = voices.length >= voiceLimit;
}

async function toggleTranscript(id) {
  const panel = $(`transcript-${id}`);
  panel.hidden = !panel.hidden;
  if (panel.hidden) return;
  const response = await fetch(`/api/voices/${id}/transcript`);
  const data = await response.json();
  panel.querySelector('textarea').value = response.ok ? data.text : '';
  panel.querySelector('.muted').textContent = response.ok ? '' : data.error;
}

async function saveTranscript(id) {
  const panel = $(`transcript-${id}`);
  const status = panel.querySelector('.muted');
  status.textContent = 'Salvataggio…';
  const response = await fetch(`/api/voices/${id}/transcript`, {
    method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: panel.querySelector('textarea').value })
  });
  const data = await response.json();
  status.textContent = response.ok ? 'Salvata: vale dalla prossima generazione.' : data.error;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}

async function loadVoices(selectedId) {
  const response = await fetch('/api/voices');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Impossibile leggere le voci salvate.');
  voices = data.voices;
  voiceLimit = data.limit || voiceLimit;
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
    stability: $('stability').value, quality: $('quality').value, speed: Number($('speed').value), refLength: $('refLength').value
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

const styleLabels = { photo: 'foto realistica', illustration: 'illustrazione', free: 'stile libero' };

function imageHistoryItem(entry, base, when) {
  const shots = entry.images.map((name, index) => `<figure><a href="${base}/${name}" target="_blank" rel="noopener"><img src="${base}/${name}" alt="" loading="lazy" /></a>
    <figcaption><span>seed ${entry.seeds[index] ?? '?'}</span><a href="${base}/${name}?download">Scarica</a></figcaption></figure>`).join('');
  const meta = [`${entry.images.length} ${entry.images.length === 1 ? 'immagine' : 'immagini'}`, entry.width ? `${entry.width}×${entry.height}` : null,
    styleLabels[entry.style] || null, entry.cost != null ? `$${entry.cost}` : null, bytes(entry.bytes)].filter(Boolean).join(' · ');
  return `<article class="history-item">
    <div class="history-head"><b>Immagini</b><span>${when}</span></div>
    <p class="history-text">${escapeHtml(entry.prompt.length > 220 ? `${entry.prompt.slice(0, 220)}…` : entry.prompt)}</p>
    <div class="history-meta">${meta}</div>
    <div class="gallery">${shots}</div>
    <div class="actions">
      <a class="download" href="${base}/immagini.zip?download">Tutte (ZIP)</a>
      <button type="button" class="ghost" data-reuse-prompt="${entry.id}">Riusa il prompt</button>
      <button type="button" class="secondary" data-delete="${entry.id}">Elimina</button>
    </div>
  </article>`;
}

function percent(value) {
  return value == null ? '–' : `${Math.round(value * 100)}%`;
}

function trainingHistoryItem(entry, base, when) {
  const players = (label) => [1, 2].map((n) => entry.samples.includes(`${label}-${n}.wav`)
    ? `<audio controls preload="none" src="${base}/${label}-${n}.wav"></audio>` : '').join('');
  const epochs = Object.keys(entry.epochScores).sort((a, b) => Number(a) - Number(b));
  const rows = [
    `<tr><td><b>La tua voce vera</b></td><td>–</td><td>riferimento</td><td>${players('vera')}</td></tr>`,
    entry.baseline ? `<tr><td>Clonazione al volo</td><td>${percent(entry.baseline.fidelity)}</td><td>${percent(entry.baseline.similarity)}</td><td>${players('clone')}</td></tr>` : '',
    ...epochs.map((n) => {
      const score = entry.epochScores[n];
      return `<tr class="${Number(n) === entry.bestEpoch ? 'best' : ''}"><td>Epoca ${n}${Number(n) === entry.bestEpoch ? ' ★ salvata' : ''}</td><td>${percent(score.fidelity)}</td><td>${percent(score.similarity)}</td><td>${players(`epoca-${String(n).padStart(2, '0')}`)}</td></tr>`;
    })
  ].join('');
  return `<article class="history-item">
    <div class="history-head"><b>Addestramento · ${escapeHtml(entry.repo || '')}</b><span>${when}</span></div>
    <div class="history-meta">${entry.clips} frasi (${entry.minutes} min) · ${entry.epochs} epoche · somiglianza misurata con WavLM rispetto alla tua voce vera</div>
    <div class="table-wrap"><table class="epochs"><thead><tr><th></th><th>Fedeltà al testo</th><th>Somiglianza a te</th><th>Ascolta</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="actions"><button type="button" class="secondary" data-delete="${entry.id}">Elimina</button></div>
  </article>`;
}

function renderHistory(allEntries) {
  const list = $('historyList');
  const entries = allEntries.filter((entry) => (currentTab === 'image' ? entry.kind === 'images' : entry.kind !== 'images'));
  if (!entries.length) {
    list.innerHTML = `<p class="muted">Ancora nessuna ${currentTab === 'image' ? 'immagine' : 'generazione vocale'}.</p>`;
    return;
  }
  list.innerHTML = entries.map((entry) => {
    const base = `/api/jobs/${entry.id}/files`;
    const when = new Date(entry.createdAt).toLocaleString('it-IT', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
    if (entry.kind === 'images') return imageHistoryItem(entry, base, when);
    if (entry.kind === 'training') return trainingHistoryItem(entry, base, when);
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
  list.querySelectorAll('[data-reuse-prompt]').forEach((button) => button.addEventListener('click', () => {
    const entry = entries.find((item) => item.id === button.dataset.reusePrompt);
    $('imagePrompt').value = entry.prompt;
    if (entry.style) $('imageStyle').value = entry.style;
    if (entry.format) $('imageFormat').value = entry.format;
    $('imagePrompt').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));
  list.querySelectorAll('[data-delete]').forEach((button) => button.addEventListener('click', async () => {
    if (!confirm('Eliminare definitivamente questa generazione dal PC?')) return;
    const response = await fetch(`/api/history/${button.dataset.delete}`, { method: 'DELETE' });
    if (!response.ok) alert((await response.json()).error || 'Non riesco a eliminarla.');
    loadHistory();
  }));
}

let historyEntries = [];
async function loadHistory() {
  try {
    const response = await fetch('/api/history');
    if (response.ok) historyEntries = (await response.json()).entries;
  } catch { /* Server closed: keep the current list. */ }
  renderHistory(historyEntries);
}

// Tabs: #voice and #image, so each section can be shown (and filmed) on its own.
let currentTab = 'voice';
function showTab() {
  currentTab = location.hash === '#image' ? 'image' : 'voice';
  $('tab-voice').hidden = currentTab !== 'voice';
  $('tab-image').hidden = currentTab !== 'image';
  $('heroWord').textContent = currentTab === 'image' ? 'Image' : 'Voice';
  $('heroPill').textContent = currentTab === 'image' ? 'Le tue immagini, senza abbonamenti' : 'Le tue voci, sul tuo PC';
  $('heroText').textContent = currentTab === 'image'
    ? 'Foto realistiche e illustrazioni con FLUX su una GPU noleggiata al minuto: descrivi l’immagine, scegli formato e varianti, scarica.'
    : 'Clona la tua voce da un campione o addestrala sulle tue registrazioni. Scegli la voce, incolla un testo e genera il tuo voice-over.';
  document.querySelectorAll('.tabs a').forEach((link) => link.classList.toggle('active', link.dataset.tab === currentTab));
  renderHistory(historyEntries);
}
window.addEventListener('hashchange', showTab);
showTab();
loadHistory();

// Image generation with FLUX.
const imageTitles = { searching: 'Ricerca GPU', creating: 'Noleggio GPU', booting: 'Avvio GPU', loading: 'Preparazione FLUX', uploading: 'Invio richiesta', generating: 'Generazione', downloading: 'Download', destroying: 'Distruzione GPU', done: 'Completato', error: 'Errore' };
let imageJob = null;

$('imageButton').addEventListener('click', async () => {
  $('imageOutput').hidden = true;
  const payload = {
    prompt: $('imagePrompt').value, style: $('imageStyle').value, format: $('imageFormat').value,
    count: Number($('imageCount').value), quality: $('imageQuality').value, seed: $('imageSeed').value,
    confirmDestroy: $('imageConfirm').checked, keepAlive: document.querySelector('input[name="imageLifecycle"]:checked').value === 'keep'
  };
  const response = await fetch('/api/images', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) return alert(data.error);
  imageJob = data.id;
  $('imageButton').disabled = true;
  $('imageCancel').hidden = false;
  $('imageProgress').hidden = false;
  try {
    for (;;) {
      const { job } = await (await fetch(`/api/jobs/${imageJob}`)).json();
      $('imageTitle').textContent = imageTitles[job.status] || job.status;
      $('imagePercent').textContent = `${Math.round(job.percent)}%`;
      $('imageBar').style.width = `${job.percent}%`;
      const cost = job.rate ? ` · ${job.gpu} · spesa $${job.cost.toFixed(3)}` : '';
      $('imageDetail').textContent = (job.status === 'error' ? `${job.error}${job.instanceId ? (job.destroyed ? ' · GPU distrutta.' : '') : ''}` : job.detail) + cost;
      if (job.status === 'done') {
        const base = `/api/jobs/${job.id}/files`;
        $('imageGallery').innerHTML = job.images.map((name, index) => `<figure><a href="${base}/${name}" target="_blank" rel="noopener"><img src="${base}/${name}" alt="" /></a>
          <figcaption><span>seed ${job.seeds[index]}</span><a href="${base}/${name}?download">Scarica</a></figcaption></figure>`).join('');
        $('imageZip').href = `${base}/immagini.zip?download`;
        $('imageInfo').textContent = `${job.images.length} immagini · ${job.width}×${job.height} · costo GPU stimato $${job.cost}`;
        $('imageOutput').hidden = false;
        loadHistory();
        break;
      }
      if (job.status === 'error') break;
      await new Promise((resolve) => setTimeout(resolve, 2500));
    }
  } finally {
    imageJob = null;
    $('imageButton').disabled = false;
    $('imageCancel').hidden = true;
  }
});

$('imageCancel').addEventListener('click', async () => {
  if (imageJob && confirm('Annullare? La GPU verrà distrutta.')) await fetch(`/api/jobs/${imageJob}/cancel`, { method: 'POST' });
});

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

// Fine-tuning: Hugging Face token, recordings and the training job.
async function loadHf() {
  const data = await (await fetch('/api/hf')).json();
  $('hfStatus').textContent = !data.configured ? 'Nessun token salvato.'
    : data.canWrite ? `Token salvato: utente ${data.user}, permesso di scrittura ✓` : `Token salvato (utente ${data.user}) ma senza permesso di scrittura: creane uno “Write”.`;
}

$('saveHf').addEventListener('click', async () => {
  $('hfStatus').textContent = 'Verifica del token…';
  const response = await fetch('/api/hf', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ token: $('hfToken').value }) });
  const data = await response.json();
  $('hfToken').value = '';
  if (!response.ok) $('hfStatus').textContent = data.error;
  else loadHf();
});

function renderTrainingFiles(files) {
  const list = $('trainingFiles');
  if (!files.length) {
    list.innerHTML = '<p class="muted">Nessuna registrazione: aggiungine almeno una (meglio 30–60 minuti in tutto).</p>';
    return;
  }
  const total = files.reduce((sum, file) => sum + file.size, 0);
  list.innerHTML = files.map((file) => `<div class="voice-row"><div><b>${escapeHtml(file.name)}</b><small>${bytes(file.size)}</small></div><button class="delete-training" data-name="${escapeHtml(file.name)}" type="button">Rimuovi</button></div>`).join('')
    + `<p class="muted">${files.length} file · ${bytes(total)} in tutto</p>`;
  list.querySelectorAll('.delete-training').forEach((button) => button.addEventListener('click', async () => {
    if (!confirm(`Rimuovere “${button.dataset.name}” dalle registrazioni di addestramento?`)) return;
    renderTrainingFiles((await (await fetch(`/api/training/files/${encodeURIComponent(button.dataset.name)}`, { method: 'DELETE' })).json()).files);
  }));
}

async function loadTrainingFiles() {
  renderTrainingFiles((await (await fetch('/api/training/files')).json()).files);
}

$('trainingUpload').addEventListener('change', async () => {
  for (const file of $('trainingUpload').files) {
    $('trainingFiles').insertAdjacentHTML('beforeend', `<p class="muted">Copia di ${escapeHtml(file.name)}…</p>`);
    const response = await fetch('/api/training/files', { method: 'POST', headers: { 'x-file-name': encodeURIComponent(file.name) }, body: file });
    const data = await response.json();
    if (!response.ok) alert(data.error);
    else renderTrainingFiles(data.files);
  }
  $('trainingUpload').value = '';
});

const trainTitles = { searching: 'Ricerca GPU', creating: 'Noleggio GPU', booting: 'Avvio GPU', loading: 'Caricamento modello', uploading: 'Invio registrazioni', training: 'Addestramento', destroying: 'Distruzione GPU', done: 'Completato', error: 'Errore' };
let trainingJob = null;

$('startTraining').addEventListener('click', async () => {
  if (!confirm('Avviare l’addestramento? Verrà noleggiata una GPU da almeno 40 GB per 30–90 minuti (circa 0,50–1,50 $), poi distrutta.')) return;
  const response = await fetch('/api/training/start', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ voiceName: $('trainedName').value, epochs: Number($('epochs').value) }) });
  const data = await response.json();
  if (!response.ok) return alert(data.error);
  trainingJob = data.id;
  $('startTraining').disabled = true;
  $('cancelTraining').hidden = false;
  $('trainProgress').hidden = false;
  try {
    for (;;) {
      const { job } = await (await fetch(`/api/jobs/${trainingJob}`)).json();
      $('trainTitle').textContent = trainTitles[job.status] || job.status;
      $('trainPercent').textContent = `${Math.round(job.percent)}%`;
      $('trainBar').style.width = `${job.percent}%`;
      const extra = [job.gpu, job.clips ? `${job.clips} frasi (${job.minutes} min)` : null, job.rate ? `spesa $${job.cost.toFixed(3)}` : null].filter(Boolean).join(' · ');
      $('trainDetail').textContent = (job.status === 'error' ? `${job.error}${job.instanceId ? (job.destroyed ? ' · GPU distrutta.' : ' · controlla cloud.vast.ai.') : ''}` : job.detail) + (extra ? ` · ${extra}` : '');
      if (job.status === 'done' || job.status === 'error') break;
      await new Promise((resolve) => setTimeout(resolve, 4000));
    }
    await loadVoices();
  } finally {
    trainingJob = null;
    $('startTraining').disabled = false;
    $('cancelTraining').hidden = true;
  }
});

$('cancelTraining').addEventListener('click', async () => {
  if (trainingJob && confirm('Annullare l’addestramento? La GPU verrà distrutta e il lavoro fatto finora andrà perso.')) {
    await fetch(`/api/jobs/${trainingJob}/cancel`, { method: 'POST' });
  }
});

loadHf().catch(() => {});
loadTrainingFiles().catch(() => {});

loadVoices().catch((error) => { $('voiceList').innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`; });
