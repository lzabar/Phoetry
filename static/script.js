const API_BASE = "/";

async function fetchModels() {
  const res = await fetch(API_BASE + "available_models");
  const data = await res.json();
  const modelSelect = document.getElementById("modelSelect") || document.getElementById("modelUsed");

  if (modelSelect) {
    modelSelect.innerHTML = "";
    for (let key in data) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.text = key;
      modelSelect.appendChild(opt);
    }
  }
}

async function fetchThemes() {
  const res = await fetch(API_BASE + "available_themes");
  const data = await res.json();
  const themeSelect = document.getElementById("themeSelect");

  if (themeSelect) {
    themeSelect.innerHTML = "";
    for (let theme of data) {
      const opt = document.createElement("option");
      opt.value = theme;
      opt.text = theme;
      themeSelect.appendChild(opt);
    }
  }
}

async function generatePoem() {
  const model = document.getElementById("modelUsed").value;
  const theme = document.getElementById("themeSelect").value;
  const fileInput = document.getElementById("fileInput");
  const formData = new FormData();

  formData.append("file", fileInput.files[0]);
  formData.append("theme", theme);
  formData.append("model_name", model);

  const res = await fetch(API_BASE + "generate_poem_from_image/", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  document.getElementById("predictedWord").innerText = data.predicted_word || "Non trouvé";
  document.getElementById("poemOutput").innerText = data.poem || "Erreur lors de la génération du poème.";
}

function previewImage() {
  const file = document.getElementById("fileInput").files[0];
  const preview = document.getElementById("preview");

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
}

// Initialisation à l’ouverture de la page
document.addEventListener("DOMContentLoaded", () => {
  fetchModels();
  fetchThemes();
  document.getElementById("fileInput").addEventListener("change", previewImage);
});
