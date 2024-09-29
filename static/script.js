document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const resultContainer = document.createElement('div');
    resultContainer.id = 'result';
    form.appendChild(resultContainer);

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const formData = new FormData(form);

        // Send the form data to the Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json(); // Assuming the server returns JSON
            const message = result.prediction === 1 
                ? "The customer is likely to churn." 
                : "The customer is not likely to churn.";
            resultContainer.innerHTML = `<p>${message}</p>`; // Display result in the div
        } else {
            const errorMessage = await response.text(); // Get error response text
            resultContainer.innerHTML = `<p>Error: ${errorMessage}</p>`; // Display error message
        }
        
    });
});
