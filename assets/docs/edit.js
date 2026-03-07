/* Inline editing toolbar. Drop <script src="edit.js"></script> before </body>. */
(function () {
  if (document.getElementById('edit-bar')) return; // prevent double init

  const SELECTOR = 'p, li, td, blockquote, figcaption';
  const SKIP = '#edit-bar, #edit-bar *';
  let editing = false;

  // build bar
  const bar = document.createElement('div');
  bar.id = 'edit-bar';
  bar.innerHTML = `
    <span id="edit-status">read-only</span>
    <button id="edit-btn">edit</button>
    <button id="save-btn" style="display:none">save</button>
    <span id="edit-hint" style="margin-left:auto"></span>`;
  Object.assign(bar.style, {
    position: 'fixed', bottom: '0', left: '0', right: '0',
    background: '#1a1a1a', color: '#ccc',
    fontFamily: "'JetBrains Mono', monospace", fontSize: '11px',
    display: 'flex', alignItems: 'center', gap: '12px',
    padding: '6px 16px', zIndex: '9999',
    borderTop: '1px solid #333',
    transition: 'opacity 0.2s', opacity: '0.4',
  });
  bar.onmouseenter = () => bar.style.opacity = '1';
  bar.onmouseleave = () => bar.style.opacity = editing ? '1' : '0.4';

  // style buttons
  bar.querySelectorAll('button').forEach(b => Object.assign(b.style, {
    background: 'none', border: '1px solid #555', color: '#eee',
    fontFamily: 'inherit', fontSize: '11px',
    padding: '2px 10px', cursor: 'pointer', borderRadius: '2px',
  }));

  document.body.appendChild(bar);

  // make links clickable even in edit mode
  document.addEventListener('click', e => {
    if (!editing) return;
    const link = e.target.closest('a');
    if (link && link.href && !link.closest('#edit-bar')) {
      window.location.href = link.href;
    }
  });

  const status = document.getElementById('edit-status');
  const btn = document.getElementById('edit-btn');
  const saveBtn = document.getElementById('save-btn');
  const hint = document.getElementById('edit-hint');

  btn.onclick = () => {
    editing = !editing;
    document.querySelectorAll(SELECTOR).forEach(el => {
      if (editing) {
        el.setAttribute('contenteditable', 'true');
        el.style.outline = '1px dashed rgba(100,150,255,0.25)';
      } else {
        el.removeAttribute('contenteditable');
        el.style.removeProperty('outline');
        if (!el.getAttribute('style')) el.removeAttribute('style');
      }
    });
    status.textContent = editing ? 'editing' : 'read-only';
    btn.textContent = editing ? 'lock' : 'edit';
    saveBtn.style.display = editing ? 'inline' : 'none';
    hint.textContent = editing ? 'click any text to edit' : '';
    bar.style.opacity = editing ? '1' : '0.4';
  };

  saveBtn.onclick = () => {
    // clone the entire document so we don't touch the live DOM
    const clone = document.documentElement.cloneNode(true);

    // strip editables from clone
    clone.querySelectorAll(SELECTOR).forEach(el => {
      el.removeAttribute('contenteditable');
      el.style.removeProperty('outline');
      if (!el.getAttribute('style')) el.removeAttribute('style');
    });

    // remove edit-bar from clone (keep script tag clean)
    const cloneBar = clone.querySelector('#edit-bar');
    if (cloneBar) cloneBar.remove();

    // find the original script src, clean it, and re-add to clone
    const origScript = document.querySelector('script[src$="edit.js"]');
    const scriptSrc = origScript ? origScript.getAttribute('src') : 'edit.js';
    clone.querySelectorAll('script[src$="edit.js"]').forEach(s => s.remove());
    const cleanScript = document.createElement('script');
    cleanScript.src = scriptSrc;
    clone.querySelector('body').appendChild(cleanScript);

    const html = '<!doctype html>\n' + clone.outerHTML;

    fetch(window.location.pathname, {
      method: 'POST',
      headers: { 'Content-Type': 'text/html' },
      body: html,
    }).then(r => {
      if (r.ok) {
        hint.textContent = 'saved';
        hint.style.color = '#8f8';
      } else {
        hint.textContent = 'save failed';
        hint.style.color = '#f88';
      }
      setTimeout(() => {
        hint.textContent = 'click any text to edit';
        hint.style.color = '#666';
      }, 2000);
    });
  };
})();
