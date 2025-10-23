---
layout: default
description: " "
redirect_from:
    - /app
---

<style>
    .embed-container {
        height: 100vh;
    }

    .embed-container iframe {
        position: absolute;
        left: 0;
        width: 100vw;
        height: 100vh;
        border: none;
    }
</style>

<div>
    <p> Web app for visualising uncertainty in climate-sensitive epidemiological
        projections (hosted on
        <a href="https://huggingface.co/spaces/will-s-hart/climepi-web-app">
            Hugging Face Spaces</a>).
        A full-page version of the app is available
        <a href="https://will-s-hart-climepi-web-app.hf.space">
            here</a>.
        Note that the web app is only intended for exploratory purposes and is not able
        to handle multiple users simultaneously. For more advanced usage, consider
        running the app locally or using the accompanying Python package (see the Python
        package
        <a href="https://climate-epidemics.readthedocs.io/en/stable">
            documentation</a>).
    </p>
    <p>
        If you notice any bugs or would like a new feature implemented, or have any
        questions about the functionality, please open an issue on
        <a href="https://github.com/idm-oxford/climate-epidemics">GitHub</a>.
        </p>
    <p> THIS APP IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
        FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
        COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
        IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
        CONNECTION WITH THE APP OR THE USE OR OTHER DEALINGS IN THE APP.
    </p>
</div>
<div class="embed-container">
    <iframe src="https://will-s-hart-climepi-web-app.hf.space"></iframe>
</div>
