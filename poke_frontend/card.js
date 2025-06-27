document.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);
  const get = (key) => params.get(key) || "Unknown";

  // Populate static info
  document.getElementById("cardName").textContent = get("name");
  document.getElementById("cardId").textContent = get("id");
  document.getElementById("cardSet").textContent = get("set");
  document.getElementById("cardArtist").textContent = get("artist");
  document.getElementById("cardRarity").textContent = get("rarity");

  const imageURL = get("image");
  document.getElementById("cardImage").src =
    imageURL !== "Unknown" && imageURL !== ""
      ? imageURL
      : "https://via.placeholder.com/300x420?text=No+Image";

  const priceEntries = [...params.entries()]
    .filter(([key, val]) => key.startsWith("cardMarketPrice-") && val !== "Unknown")
    .map(([key, val]) => ({
      date: key.replace("cardMarketPrice-", ""),
      price: parseFloat(val),
    }))
    .filter((entry) => !isNaN(entry.price))
    .sort((a, b) => {
      const toDate = (str) => new Date("20" + str.replace(/-/g, "/"));
      return toDate(a.date) - toDate(b.date);
    });

  const priceHeader = document.getElementById("priceHeader");
  const chartCanvas = document.getElementById("priceChart");

  if (priceEntries.length === 0) {
    priceHeader.textContent = "No card price data is available.";
    chartCanvas.style.display = "none";
    return;
  }

  const labels = priceEntries.map(
    (e) => `2025-${e.date.slice(0, 2)}-${e.date.slice(3, 5)}`
  );
  const prices = priceEntries.map((e) => e.price);
  const latest = prices[prices.length - 1];
  const previous = prices.length > 1 ? prices[prices.length - 2] : latest;
  const change = latest - previous;
  const changePercent = previous !== 0 ? (change / previous) * 100 : 0;

  priceHeader.textContent = `Current Price: $${latest.toFixed(2)}`;

  const pointColors = prices.map((price, i, arr) => {
    if (i === 0) return "#4eaaff";
    if (price > arr[i - 1]) return "green";
    if (price < arr[i - 1]) return "red";
    return "#4eaaff";
  });

  const segmentColors = prices.map((price, i, arr) => {
    if (i === 0) return null;
    if (price > arr[i - 1]) return "green";
    if (price < arr[i - 1]) return "red";
    return "#4eaaff";
  });

  new Chart(chartCanvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Price",
          data: prices,
          borderColor: (ctx) => segmentColors[ctx.p1DataIndex] || "#4eaaff",
          backgroundColor: "rgba(78,170,255,0.2)",
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: pointColors,
          tension: 0.3,
          fill: true,
          segment: {
            borderColor: (ctx) => segmentColors[ctx.p1DataIndex] || "#4eaaff",
          },
        },
      ],
    },
    options: {
      scales: {
        x: { ticks: { color: "#ccc" } },
        y: { beginAtZero: false, ticks: { color: "#ccc" } },
      },
      plugins: {
        legend: { labels: { color: "#ccc" } },
      },
    },
  });

  // 30-Day Summary Table
  const table = document.createElement("table");
  table.style.marginTop = "20px";
  table.style.color = "#ccc";
  table.style.borderCollapse = "collapse";

  const today = new Date();
  const cutoff = new Date(today);
  cutoff.setDate(today.getDate() - 30);

  const filtered = priceEntries.filter((entry) => {
    const [mm, dd, yy] = entry.date.split("-");
    const parsedDate = new Date(`20${yy}-${mm}-${dd}`);
    return parsedDate >= cutoff;
  });

  let low = "N/A", high = "N/A";
  if (filtered.length > 0) {
    const pricesOnly = filtered.map((p) => p.price);
    low = `$${Math.min(...pricesOnly).toFixed(2)}`;
    high = `$${Math.max(...pricesOnly).toFixed(2)}`;
  }

  const changeClass = change > 0 ? 'change-positive' : change < 0 ? 'change-negative' : '';

  table.innerHTML = `
    <tr>
      <th style="padding: 8px; border-bottom: 1px solid #666;">30-Day Low</th>
      <td style="padding: 8px; border-bottom: 1px solid #666;">${low}</td>
    </tr>
    <tr>
      <th style="padding: 8px; border-bottom: 1px solid #666;">30-Day High</th>
      <td style="padding: 8px; border-bottom: 1px solid #666;">${high}</td>
    </tr>
    <tr>
      <th style="padding: 8px; border-bottom: 1px solid #666;">Change Since Yesterday</th>
      <td style="padding: 8px; border-bottom: 1px solid #666;" class="${changeClass}">
        ${change >= 0 ? "+" : ""}${change.toFixed(2)} 
        (${changePercent >= 0 ? "+" : ""}${changePercent.toFixed(2)}%)
      </td>
    </tr>
  `;

  chartCanvas.parentElement.appendChild(table);
});
