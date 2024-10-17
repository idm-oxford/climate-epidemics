---
layout: default
---

Python package documentation: [https://climate-epidemics.readthedocs.io/en/latest/](https://climate-epidemics.readthedocs.io/en/latest/)

Web app:

<div id="loading-message">
    The app is waking up, please wait a moment...
</div>

<iframe id="app-frame" src="https://will-s-hart-climepi-web-app.hf.space" allowfullscreen></iframe>

<div class="spacer"></div>

<script>
    // Hide loading message once iframe loads
    const iframe = document.getElementById('app-frame');
    iframe.onload = function() {
        document.getElementById('loading-message').style.display = 'none';
    };
</script>
