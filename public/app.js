const $ = (id) => document.getElementById(id);
const reference = $('reference');
const text = $('text');
const result = $('result');
let profileReady = false;

function updateCharacters() {
  $('characters').textContent = `${text.value.length.toLocaleString('it-IT')} caratteri`;
}

text.addEventListener('input', updateCharacters);
updateCharacters();

reference.addEventListener('change', () => {
  profileReady = false;
  $('generateCard').hidden = true;
  const file = reference.files[0];
  $('referenceStatus').textContent = file
    ? `${file.name} selezionato. Ora premi “Crea la mia voce”.`
    : 'Nessun file selezionato.';
});

$('cloneButton').addEventListener('click', () => {
  const file = reference.files[0];
  const status = $('profileStatus');
  if (!file) {
    status.textContent = 'Prima scegli un file audio della tua voce.';
    return;
  }
  profileReady = true;
  $('generateCard').hidden = false;
  status.textContent = '✓ La tua voce è pronta per questa sessione. Ora scrivi cosa deve dire.';
  $('generateCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

$('generateButton').addEventListener('click', async () => {
  if (!profileReady) return;
  result.hidden = false;
  result.innerHTML = '<p>Controllo budget e impostazioni...</p>';
  const payload = {
    characters: text.value.length,
    hourlyRate: Number($('rate').value),
    minutes: Number($('minutes').value),
    safetyMinutes: 10,
    budgetCap: Number($('budget').value),
    referenceReady: Boolean(reference.files[0]),
    referenceTranscriptReady: true,
    confirmDestroy: $('destroy').checked
  };
  try {
    const response = await fetch('/api/job-plan', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
      result.innerHTML = `<h2>Controlla il job</h2><p>${data.error}</p>`;
      return;
    }
    result.innerHTML = `<h2>Profilo pronto, piano approvato</h2><div class="cost"><b>$${data.cost.estimatedGpuCost}</b><span>stima GPU per ${data.cost.totalMinutes} minuti</span></div><p>La parte che noleggia Vast e restituisce il WAV deve ancora essere collegata. Non è stato avviato alcun addebito.</p>`;
  } catch {
    result.innerHTML = '<h2>Server non raggiungibile</h2><p>Chiudi e riapri START.bat, poi ricarica questa pagina.</p>';
  }
});
