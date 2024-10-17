---
layout: default
title: climepi home
---

Python package documentation: [https://climate-epidemics.readthedocs.io/en/latest/](https://climate-epidemics.readthedocs.io/en/latest/)

Web app:

<div id="loading-message">
    The app is waking up, please wait a moment...
</div>

<iframe id="app-frame" src="https://will-s-hart-climepi-web-app.hf.space" allowfullscreen></iframe>

<script>
    // Hide loading message once the iframe is ready
    const iframe = document.getElementById('app-frame');
    iframe.onload = function() {
        document.getElementById('loading-message').style.display = 'none';
    };
</script>