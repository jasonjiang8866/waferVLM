'use strict';

const collections = JSON.parse(document.getElementById('catalog').textContent);
const byId = id => document.getElementById(id);
const dayMs = 86_400_000;
const singaporeOffset = 8 * 3_600_000;
const dialog = byId('detail');
let edition;
let timer;
let seen = new Set();

function dayNumber(now) {
  return Math.floor((now + singaporeOffset) / dayMs);
}

function selection(collection, day) {
  const offset = Array.from(collection.id).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return collection.images[(day + offset) % collection.images.length];
}

function updateProgress() {
  document.querySelectorAll('.card').forEach(card => {
    const opened = seen.has(card.dataset.collection);
    card.dataset.seen = opened;
    card.querySelector('.open').textContent = opened ? '✓' : '+';
  });
  const count = collections.filter(collection => seen.has(collection.id)).length;
  byId('bar').max = collections.length;
  byId('bar').value = count;
  byId('score').textContent = count + ' / ' + collections.length;
  byId('status').textContent = count === collections.length ? 'All datasets explored today.' : count + (count === 1 ? ' dataset' : ' datasets') + ' explored on this device';
}

function setImage(img, row) {
  const frame = img.parentElement;
  const fallback = frame.nextElementSibling;
  frame.classList.toggle('cropped', Boolean(row.crop));
  img.removeAttribute('style');
  if (row.crop) {
    const [x, y, width, height, sourceWidth, sourceHeight] = row.crop;
    img.style.width = (100 * sourceWidth / width) + '%';
    img.style.height = (100 * sourceHeight / height) + '%';
    img.style.left = (-100 * x / width) + '%';
    img.style.top = (-100 * y / height) + '%';
  }
  frame.hidden = false;
  img.hidden = false;
  fallback.hidden = true;
  img.onerror = () => { frame.hidden = true; fallback.hidden = false; };
  img.referrerPolicy = 'no-referrer';
  img.alt = row.title;
  img.src = row.src;
}

function inspect(collection, row) {
  if (edition !== dayNumber(Date.now())) { refresh(); return; }
  byId('modal-type').textContent = collection.type;
  byId('modal-name').textContent = collection.name;
  byId('modal-title').textContent = row.title;
  byId('question').textContent = collection.question;
  byId('note').textContent = row.note || '';
  byId('credit').textContent = row.credit;
  byId('source').href = row.source;
  byId('full-link').href = row.src;
  byId('license').hidden = !row.license;
  if (row.license) byId('license').href = row.license;
  setImage(byId('full'), row);
  dialog.showModal();
  seen.add(collection.id);
  try { localStorage.setItem('wafer-daily-progress', JSON.stringify({edition, seen: [...seen]})); } catch {}
  updateProgress();
}

function refresh() {
  const now = Date.now();
  const day = dayNumber(now);
  if (day !== edition) {
    if (dialog.open) dialog.close();
    edition = day;
    seen = new Set();
    try {
      const saved = JSON.parse(localStorage.getItem('wafer-daily-progress'));
      if (saved?.edition === edition && Array.isArray(saved.seen)) seen = new Set(saved.seen);
    } catch {}
    const date = new Date(day * dayMs);
    byId('date').dateTime = date.toISOString().slice(0, 10);
    byId('date').textContent = date.toLocaleDateString('en-GB', {timeZone:'UTC', weekday:'short', day:'numeric', month:'short', year:'numeric'});
    byId('total').textContent = collections.length + ' datasets';
    byId('gallery').replaceChildren(...collections.map(collection => {
      const row = selection(collection, day);
      const card = document.createElement('button');
      card.className = 'card';
      card.dataset.collection = collection.id;
      card.setAttribute('aria-haspopup', 'dialog');
      card.setAttribute('aria-label', 'Inspect ' + collection.name + ' — ' + row.title);
      card.innerHTML = '<span class="stage"><span class="image-frame"><img alt=""></span><span class="fallback" hidden>Preview unavailable · View source</span></span><span class="caption"><span class="type"></span><strong class="name"></strong><span class="open" aria-hidden="true">+</span></span>';
      card.querySelector('.type').textContent = collection.type;
      card.querySelector('.name').textContent = collection.name;
      setImage(card.querySelector('img'), row);
      card.onclick = () => inspect(collection, row);
      return card;
    }));
    updateProgress();
  }
  clearTimeout(timer);
  timer = setTimeout(refresh, (day + 1) * dayMs - singaporeOffset - now + 20);
}

byId('close').onclick = () => dialog.close();
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
window.addEventListener('pageshow', refresh);
refresh();
