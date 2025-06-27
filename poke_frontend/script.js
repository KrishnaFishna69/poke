let cardData = [];
let currentIndex = -1;

document.addEventListener("DOMContentLoaded", () => {
  Papa.parse("data.csv", {
    download: true,
    header: true,
    complete: function(results) {
      cardData = results.data;

      // Background animation initialization
      const images = cardData
        .map(card => card.images)
        .filter(src => src && src.startsWith("http"));
      addFloatingCards(images);
    }
  });

  const searchInput = document.getElementById("searchInput");
  const suggestionsBox = document.getElementById("autocomplete-list");

  searchInput.addEventListener("input", () => {
    const rawTerm = searchInput.value.toLowerCase();
    suggestionsBox.innerHTML = "";
    currentIndex = -1;

    if (!rawTerm) {
      suggestionsBox.style.display = "none";
      return;
    }

    let nameQuery = rawTerm;
    let setQuery = null;

    if (rawTerm.includes(";")) {
      [nameQuery, setQuery] = rawTerm.split(";").map(s => s.trim());
    }

    const matches = cardData.filter(card => {
      const nameMatch = card.name?.toLowerCase().startsWith(nameQuery);
      if (!nameMatch) return false;

      if (setQuery) {
        const match = card.set?.match(/name='([^']+)'/);
        const setName = match ? match[1].toLowerCase() : "";
        return setName.includes(setQuery);
      }

      return true;
    });

    matches.forEach(card => {
      const item = document.createElement("div");
      item.className = "autocomplete-item";

      const img = document.createElement("img");
      img.src = card.images || "https://via.placeholder.com/40";

      const name = document.createElement("span");
      name.className = "card-name";
      name.textContent = card.name || "Unknown Name";

      const set = document.createElement("span");
      set.className = "card-set";
      const match = card.set?.match(/name='([^']+)'/);
      const setName = match ? match[1] : "Unknown Set";
      set.textContent = setName;

      item.appendChild(img);
      item.appendChild(name);
      item.appendChild(set);

      item.addEventListener("click", () => {
        const getField = (key) => {
          const val = card[key];
          return val !== undefined && val !== "Unknown" ? val : "";
        };

        const priceParams = {};
        for (const key in card) {
          if (key.startsWith("cardMarketPrice-")) {
            priceParams[key] = getField(key);
          }
        }

        const params = new URLSearchParams({
          name: getField("name"),
          id: getField("id"),
          set: setName,
          artist: getField("artist"),
          rarity: getField("rarity"),
          image: getField("images"),
          cardMarketPrice: getField("cardMarketPrice-06-16-25"), // Update if dynamic key needed
          ...priceParams
        });

        window.location.href = `card.html?${params.toString()}`;
      });

      suggestionsBox.appendChild(item);
    });

    suggestionsBox.style.display = matches.length > 0 ? "block" : "none";
  });

  searchInput.addEventListener("keydown", (e) => {
    const items = suggestionsBox.getElementsByClassName("autocomplete-item");

    if (e.key === "ArrowDown") {
      currentIndex = (currentIndex + 1) % items.length;
      updateActive(items);
    } else if (e.key === "ArrowUp") {
      currentIndex = (currentIndex - 1 + items.length) % items.length;
      updateActive(items);
    } else if (e.key === "Enter" && currentIndex > -1) {
      items[currentIndex].click();
      e.preventDefault();
    }
  });

  function updateActive(items) {
    for (let i = 0; i < items.length; i++) {
      items[i].classList.remove("active");
    }
    if (items[currentIndex]) {
      items[currentIndex].classList.add("active");
      items[currentIndex].scrollIntoView({ block: "nearest" });
    }
  }
});

// --- Floating Background Card Images ---
function addFloatingCards(imageUrls) {
  const container = document.getElementById("background-container");
  for (let i = 0; i < 20; i++) {
    const img = document.createElement("img");
    img.src = imageUrls[Math.floor(Math.random() * imageUrls.length)];
    img.className = "floating-card";
    img.style.left = `${Math.random() * 100}vw`;
    img.style.animationDuration = `${20 + Math.random() * 20}s`;
    container.appendChild(img);
  }
}
