const form = document.getElementById("predictForm");
const resetBtn = document.getElementById("resetBtn");
const submitBtn = document.getElementById("submitBtn");
const loadingSpinner = document.getElementById("loadingSpinner");
const btnText = document.querySelector(".btn-text");

const resultCard = document.getElementById("resultCard");
const probabilityText = document.getElementById("probabilityText");
const labelBadge = document.getElementById("labelBadge");
const modelUsedText = document.getElementById("modelUsedText");
const suggestionsList = document.getElementById("suggestionsList");
const alertBox = document.getElementById("alertBox");
const modelChip = document.getElementById("modelChip");
const downloadReportBtn = document.getElementById("downloadReportBtn");
const PREDICT_API_URL = "/predict";
const INSIGHTS_API_URL = "/api/model-insights";

let featureChart = null;
let probabilityPie = null;
let histogramChart = null;

function showAlert(message, type = "danger") {
  alertBox.className = `alert alert-${type}`;
  alertBox.textContent = message;
  alertBox.classList.remove("d-none");
}

function clearAlert() {
  alertBox.classList.add("d-none");
  alertBox.textContent = "";
}

function validateInputs(data) {
  if (data.cgpa < 0 || data.cgpa > 10) {
    return "CGPA must be between 0 and 10.";
  }
  if (data.skills < 0 || data.projects < 0) {
    return "Skills and projects must be non-negative.";
  }
  if (data.communication < 1 || data.communication > 10) {
    return "Communication must be between 1 and 10.";
  }
  return null;
}

function drawFeatureImportance(importanceMap) {
  const labels = Object.keys(importanceMap);
  const values = labels.map((key) => Number(importanceMap[key]) * 100);

  if (featureChart) {
    featureChart.destroy();
  }

  featureChart = new Chart(document.getElementById("featureImportanceChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Importance (%)",
          data: values,
          backgroundColor: ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
          borderRadius: 8,
        },
      ],
    },
    options: {
      animation: { duration: 900 },
      plugins: { legend: { display: false } },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: (value) => `${value}%`,
          },
        },
      },
    },
  });
}

function drawProbabilityPie(placed, notPlaced) {
  if (probabilityPie) {
    probabilityPie.destroy();
  }

  probabilityPie = new Chart(document.getElementById("probabilityPieChart"), {
    type: "pie",
    data: {
      labels: ["Placed", "Not Placed"],
      datasets: [
        {
          data: [placed, notPlaced],
          backgroundColor: ["#2b9348", "#adb5bd"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      animation: { duration: 700 },
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function drawCgpaHistogram(labels, values) {
  if (histogramChart) {
    histogramChart.destroy();
  }

  histogramChart = new Chart(document.getElementById("cgpaHistogramChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Students",
          data: values,
          backgroundColor: "rgba(69, 123, 157, 0.75)",
          borderColor: "#1d3557",
          borderWidth: 1,
        },
      ],
    },
    options: {
      animation: { duration: 900 },
      plugins: { legend: { display: false } },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

async function loadModelInsights() {
  const response = await fetch(INSIGHTS_API_URL);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Failed to load model insights.");
  }

  modelChip.textContent = `Model: ${data.selected_model}`;
  drawFeatureImportance(data.feature_importance || {});
  drawCgpaHistogram(data.cgpa_histogram.labels || [], data.cgpa_histogram.values || []);
}

async function submitPrediction(payload) {
  const response = await fetch(PREDICT_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "Prediction failed.");
  }
  return result;
}

function updateResultCard(result) {
  probabilityText.textContent = `${result.probability}%`;
  labelBadge.textContent = result.label;
  labelBadge.className = `badge text-bg-${result.badge}`;
  modelUsedText.textContent = `Predicted using: ${result.selected_model}`;

  suggestionsList.innerHTML = "";
  result.suggestions.forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    suggestionsList.appendChild(li);
  });

  drawFeatureImportance(result.feature_importance || {});
  drawProbabilityPie(
    Number(result.probability_breakdown?.placed || 0),
    Number(result.probability_breakdown?.not_placed || 0)
  );

  if (result.prediction_id) {
    downloadReportBtn.href = `/report/${result.prediction_id}`;
    downloadReportBtn.classList.remove("d-none");
  }

  resultCard.classList.remove("d-none");
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearAlert();

    const payload = {
      cgpa: Number.parseFloat(document.getElementById("cgpa").value),
      skills: Number.parseInt(document.getElementById("skills").value, 10),
      internship: document.getElementById("internship").value,
      projects: Number.parseInt(document.getElementById("projects").value, 10),
      communication: Number.parseInt(document.getElementById("communication").value, 10),
    };

    const validationError = validateInputs(payload);
    if (validationError) {
      showAlert(validationError, "danger");
      return;
    }

    submitBtn.disabled = true;
    btnText.textContent = "Predicting";
    loadingSpinner.classList.remove("d-none");

    try {
      const result = await submitPrediction(payload);
      updateResultCard(result);
      showAlert("Prediction completed successfully.", "success");
    } catch (error) {
      showAlert(error.message || "Something went wrong.", "danger");
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = "Predict";
      loadingSpinner.classList.add("d-none");
    }
  });
}

if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    form.reset();
    clearAlert();
    resultCard.classList.add("d-none");
    suggestionsList.innerHTML = "";
    downloadReportBtn.classList.add("d-none");
  });
}

loadModelInsights().catch((error) => {
  showAlert(error.message || "Unable to load model charts.", "warning");
});
