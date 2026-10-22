document.addEventListener('DOMContentLoaded', () => {
  const burger = document.querySelector('.navbar-burger');
  const menu = document.querySelector('.navbar-menu');
  function toggleMenu() {
    const expanded = burger.getAttribute('aria-expanded') !== 'true';
    burger.setAttribute('aria-expanded', String(expanded));
    burger.classList.toggle('is-active', expanded);
    menu.classList.toggle('is-active', expanded);
  }
  burger?.addEventListener('click', toggleMenu);
  burger?.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleMenu();
    }
  });

  const tabs = Array.from(document.querySelectorAll('[data-demo-tab]'));
  const panels = Array.from(document.querySelectorAll('.demo-panel'));
  const tablist = document.querySelector('.demo-tabs');
  if (tablist) tablist.setAttribute('role', 'tablist');
  tabs.forEach(tab => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', `demo-${tab.dataset.demoTab}`);
  });
  panels.forEach(panel => {
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', panel.id.replace('demo-', 'tab-'));
    panel.tabIndex = 0;
  });
  function activate(key, focus = false, updateHash = false) {
    tabs.forEach(tab => {
      const selected = tab.dataset.demoTab === key;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    });
    panels.forEach(panel => { panel.hidden = panel.id !== `demo-${key}`; });
    if (updateHash) history.replaceState(null, '', `#demo-${key}`);
  }
  function fromHash() {
    const key = location.hash.replace('#demo-', '');
    activate(tabs.some(tab => tab.dataset.demoTab === key) ? key : 't2i');
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', event => {
      event.preventDefault();
      activate(tab.dataset.demoTab, false, true);
    });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        activate(tabs[next].dataset.demoTab, true, true);
      }
    });
  });
  if (tabs.length) fromHash();
  window.addEventListener('hashchange', () => {
    if (location.hash.startsWith('#demo-')) fromHash();
  });

  document.querySelectorAll('.compare-control input').forEach(slider => {
    slider.addEventListener('input', () => {
      const compare = slider.closest('.edit-card').querySelector('.compare');
      compare.style.setProperty('--split', `${slider.value}%`);
      slider.setAttribute('aria-valuetext', `${slider.value}% input visible`);
      compare.querySelector('.input-label').hidden = Number(slider.value) < 12;
      compare.querySelector('.output-label').hidden = Number(slider.value) > 88;
    });
  });
  document.querySelectorAll('.grounding-toggle input').forEach(toggle => {
    toggle.addEventListener('change', () => {
      toggle.closest('.grounding-card').querySelector('svg').style.visibility = toggle.checked ? 'visible' : 'hidden';
    });
  });
});
