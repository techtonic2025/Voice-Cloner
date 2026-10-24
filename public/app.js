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
    list.innerHTML = voices.map((voice) => `<div class="voice-row"><div><b>${escapeHtml(voice.name)}</b><small>Profilo locale · ${bytes(voice.size)}</small></div><button class="delete-voice" data-id="${voice.id}" type="button">Elimina</button></div>`).join('');
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

$('generateButton').addEventListener('click', async () => {
  const voiceId = $('voiceSelect').value;
  $('result').hidden = false;
  $('result').innerHTML = '<p>Controllo budget e impostazioni…</p>';
  const payload = {
    voiceId, text: text.value, language: $('language').value, characters: text.value.length,
    hourlyRate: Number($('rate').value), minutes: Number($('minutes').value), safetyMinutes: 10,
    budgetCap: Number($('budget').value), confirmDestroy: $('destroy').checked
  };
  try {
    const response = await fetch('/api/job-plan', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) {
      $('result').innerHTML = `<h2>Controlla il job</h2><p>${data.error}</p>`;
      return;
    }
    $('result').innerHTML = `<h2>Piano approvato</h2><div class="cost"><b>$${data.cost.estimatedGpuCost}</b><span>stima GPU per ${data.cost.totalMinutes} minuti</span></div><p>${data.warning}</p>`;
  } catch {
    $('result').innerHTML = '<h2>Server non raggiungibile</h2><p>Chiudi e riapri START.bat, poi ricarica questa pagina.</p>';
  }
});

loadVoices().catch((error) => { $('voiceList').innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`; });
