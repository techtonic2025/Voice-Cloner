const $ = (id) => document.getElementById(id);
const text = $('text');
const chars = $('characters');
const ref = $('reference');
const result = $('result');

function updateCharacters() {
  chars.textContent = `${text.value.length.toLocaleString('it-IT')} caratteri`;
}
text.addEventListener('input', updateCharacters);
updateCharacters();

ref.addEventListener('change', () => {
  $('referenceStatus').textContent = ref.files[0] ? `${ref.files[0].name} pronto per il job (non viene caricato in questa build).` : 'Nessun file selezionato.';
});

$('voiceForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  result.hidden = false;
  result.innerHTML = '<p>Controllo costo e vincoli...</p>';
  const payload = {
    characters: text.value.length,
    hourlyRate: Number($('rate').value),
    minutes: Number($('minutes').value),
    safetyMinutes: 10,
    budgetCap: Number($('budget').value),
    referenceReady: Boolean(ref.files[0]),
    referenceTranscriptReady: Boolean($('referenceText').value.trim()),
    confirmDestroy: $('destroy').checked
  };
  const response = await fetch('/api/job-plan', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) {
    result.innerHTML = `<h2>Job non avviabile</h2><p>${data.error}</p>`;
    return;
  }
  result.innerHTML = `<h2>Piano sicuro</h2><div class="cost"><b>$${data.cost.estimatedGpuCost}</b><span>stima GPU per ${data.cost.totalMinutes} minuti</span></div><p>Tetto prudente consigliato: <b>$${data.cost.recommendedBudget}</b>.</p><ol>${data.plan.map((item) => `<li>${item}</li>`).join('')}</ol><p class="muted">${data.warning}</p>`;
});
