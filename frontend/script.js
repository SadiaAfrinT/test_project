document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('image-input');
    const imageFileInput = document.getElementById('image-file-input');
    const imagePath = imageInput.querySelector('.path');

    const videoInput = document.getElementById('video-input');
    const videoFileInput = document.getElementById('video-file-input');
    const videoPath = videoInput.querySelector('.path');

    const runAnalysisButton = document.getElementById('run-analysis');
    const detectedObjectsOutput = document.getElementById('detected-objects-output');
    const downloadObjectsButton = document.getElementById('download-objects-button');
    const downloadCsvButton = document.getElementById('download-csv-button');

    const precisionValue = document.getElementById('precision-value');
    const recallValue = document.getElementById('recall-value');
    const f1ScoreValue = document.getElementById('f1-score-value');

    let selectedFile = null;

    imageInput.addEventListener('click', () => imageFileInput.click());
    videoInput.addEventListener('click', () => videoFileInput.click());

    imageFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            imagePath.textContent = file.name;
            videoPath.textContent = 'No file chosen...';
            selectedFile = file;
        }
    });

    videoFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            videoPath.textContent = file.name;
            imagePath.textContent = 'No file chosen...';
            selectedFile = file;
        }
    });

    runAnalysisButton.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please select an image or video file first.');
            return;
        }

        runAnalysisButton.textContent = 'Analyzing...';
        runAnalysisButton.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            displayResults(results);

        } catch (error) {
            console.error('Error during analysis:', error);
            detectedObjectsOutput.innerHTML = `<p style="color: red; text-align: center;">An error occurred during analysis.</p>`;
        } finally {
            runAnalysisButton.textContent = 'Run Analysis';
            runAnalysisButton.disabled = false;
        }
    });

    function displayResults(results) {
        // Display detected objects
        detectedObjectsOutput.innerHTML = '';
        if (results.detected_objects && results.detected_objects.length > 0) {
            const list = document.createElement('ul');
            results.detected_objects.forEach(obj => {
                const item = document.createElement('li');
                item.textContent = `${obj.class} (Confidence: ${obj.confidence.toFixed(2)})`;
                list.appendChild(item);
            });
            detectedObjectsOutput.appendChild(list);
        } else {
            detectedObjectsOutput.innerHTML = `<p style="color: var(--secondary-color); text-align: center;">No objects detected.</p>`;
        }

        // Display evaluation metrics
        precisionValue.textContent = results.evaluation.precision.toFixed(2);
        recallValue.textContent = results.evaluation.recall.toFixed(2);
        f1ScoreValue.textContent = results.evaluation['f1-score'].toFixed(2);

        // Enable download buttons
        downloadObjectsButton.classList.remove('disabled');
        downloadCsvButton.classList.remove('disabled');

        // Store results for download
        downloadObjectsButton.onclick = () => downloadJSON(results.detected_objects, 'detected_objects.json');
        downloadCsvButton.onclick = () => downloadCSV(results.evaluation, 'evaluation_report.csv');
    }

    function downloadJSON(data, filename) {
        const jsonStr = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function downloadCSV(data, filename) {
        const headers = ['metric', 'value'];
        const rows = Object.entries(data).map(([key, value]) => [key, value.toFixed(4)]);
        let csvContent = "data:text/csv;charset=utf-8," 
            + headers.join(",") + "\n" 
            + rows.map(e => e.join(",")).join("\n");
        
        const encodedUri = encodeURI(csvContent);
        const a = document.createElement('a');
        a.href = encodedUri;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
});
