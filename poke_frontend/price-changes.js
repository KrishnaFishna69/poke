Papa.parse("data.csv", {
  download: true,
  header: true,
  skipEmptyLines: true,
  complete: function (results) {
    const data = results.data;
    const priceCols = Object.keys(data[0]).filter(key => key.startsWith("cardMarketPrice-"));

    if (priceCols.length < 2) {
      console.error("Not enough price columns to calculate changes.");
      return;
    }

    // Sort date columns chronologically
    priceCols.sort((a, b) => {
      const dateA = new Date("20" + a.split("-").slice(1).join("-"));
      const dateB = new Date("20" + b.split("-").slice(1).join("-"));
      return dateA - dateB;
    });

    const prevCol = priceCols[priceCols.length - 2];
    const currCol = priceCols[priceCols.length - 1];

    const changes = [];

    data.forEach((row) => {
      const name = row["name"];
      const image = row["images"];
      const setRaw = row["set"];
      const oldPrice = parseFloat(row[prevCol]);
      const newPrice = parseFloat(row[currCol]);

      if (!isNaN(oldPrice) && !isNaN(newPrice)) {
        const change = newPrice - oldPrice;
        const percentChange = (change / oldPrice) * 100;

        // Extract set name
        let set = "Unknown Set";
        const match = setRaw && setRaw.match(/name='([^']+)'/);
        if (match) set = match[1];

        changes.push({
          name,
          image,
          set,
          oldPrice,
          newPrice,
          change,
          percentChange
        });
      }
    });

    changes.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

    const tbody = document.querySelector("#priceChangeTable tbody");
    tbody.innerHTML = "";

    changes.slice(0, 50).forEach((card) => {
      const row = document.createElement("tr");

      const changeColor = card.change >= 0 ? "style='color: #00ff88;'" : "style='color: #ff4d4d;'";
      const percentColor = card.percentChange >= 0 ? "style='color: #00ff88;'" : "style='color: #ff4d4d;'";

      row.innerHTML = `
        <td><img src="${card.image}" width="60" alt="${card.name}"></td>
        <td>${card.name}</td>
        <td>${card.set}</td>
        <td>$${card.oldPrice.toFixed(2)}</td>
        <td>$${card.newPrice.toFixed(2)}</td>
        <td ${changeColor}>${card.change >= 0 ? '+' : ''}$${card.change.toFixed(2)}</td>
        <td ${percentColor}>${card.percentChange >= 0 ? '+' : ''}${card.percentChange.toFixed(2)}%</td>
      `;
      tbody.appendChild(row);
    });
  },
  error: function (err) {
    console.error("❌ PapaParse Error:", err);
  }
});
