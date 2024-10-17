Python package and front-end web application for incorporating climate data into epidemiological models.

Python package documentation: [https://climate-epidemics.readthedocs.io/en/latest/](https://climate-epidemics.readthedocs.io/en/latest/)

Web app:

<div id="loading-message">
    The app is waking up, please wait a moment...
</div>

<div class="embed-container">
    <iframe id="app-frame" src="https://will-s-hart-climepi-web-app.hf.space" allowfullscreen></iframe>
</div>

<style>
    /* Ensure the body and HTML take the full width and height of the viewport */
    body, html {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden; /* Prevent scrolling */
    }

    /* Full width and height container for the iframe */
    .embed-container {
        position: relative;
        width: 100vw;  /* Full viewport width */
        height: 100vh; /* Full viewport height */
        background-color: #f0f0f0;
    }

    /* The iframe should stretch to fill the container */
    .embed-container iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: none;
    }

    /* Style for the loading message */
    #loading-message {
        text-align: center;
        padding: 20px;
        font-size: 18px;
        background-color: #f0f0f0;
        color: #333;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 10;
        display: block;
    }
</style>

<script>
    // Hide the loading message when the iframe finishes loading
    const iframe = document.getElementById('app-frame');
    iframe.onload = function() {
        document.getElementById('loading-message').style.display = 'none';
    };
</script>
