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
    /* Reset margins and padding across all elements */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Ensure the page takes up the full viewport */
    html, body {
        width: 100vw;
        height: 100vh;
        margin: 0;
        padding: 0;
        overflow: hidden; /* Prevent default scrolling */
    }

    /* Ensure that the iframe container takes the full viewport */
    .embed-container {
        position: relative;
        width: 100vw;
        height: 100vh;
        background-color: #f0f0f0;
        overflow: hidden;
    }

    /* Fullscreen iframe with no borders */
    .embed-container iframe {
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
    }

    /* Allow scrolling when content overflows */
    .embed-container, #app-frame {
        overflow: auto;
    }
</style>

<script>
    // Hide the loading message once the iframe has loaded
    const iframe = document.getElementById('app-frame');
    iframe.onload = function() {
        document.getElementById('loading-message').style.display = 'none';
    };
</script>
