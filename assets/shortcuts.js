(function() {
    // Attach only once
    if (window._eeg_shortcuts_attached) return;
    window._eeg_shortcuts_attached = true;

    document.addEventListener('keydown', function(e) {
        // If focus is on input, skip (so user can type)
        var el = document.activeElement;
        if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
        
        // arrows -> navigate (если добавите кнопки позже)
        if (e.key === 'ArrowLeft') {
            var btn = document.getElementById('seek-back-btn');
            if (btn) btn.click();
        }
        if (e.key === 'ArrowRight') {
            var btn = document.getElementById('seek-forward-btn');
            if (btn) btn.click();
        }
    });
})();