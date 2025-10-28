document.getElementById("uploadBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("imageInput");
    const previewSection = document.getElementById("previewSection");
    const previewImage = document.getElementById("previewImage");
    const captionText = document.getElementById("captionText");

    if (!fileInput.files.length) {
        alert("Please select an image first!");
        return;
    }

    const file = fileInput.files[0];

    // Show preview image immediately
    previewSection.classList.remove("hidden");
    previewImage.src = URL.createObjectURL(file);
    captionText.textContent = "Generating caption... Please wait ⏳";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload-image/", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        captionText.textContent = data.caption || "No caption generated.";
    } catch (error) {
        captionText.textContent = "Error generating caption.";
        console.error(error);
    }
});
