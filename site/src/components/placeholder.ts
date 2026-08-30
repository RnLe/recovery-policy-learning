// Slots for hand-drawn illustrations. The page declares the slot and what
// belongs in it; when the SVG lands in public/illustrations/, the card swaps
// itself for the real figure on the next load; no build step involved.

import { publicUrl } from "../data/paths";

export function mountIllustrations(): void {
  for (const slot of document.querySelectorAll<HTMLElement>(
    "[data-illustration]",
  )) {
    const name = slot.dataset.illustration ?? "";
    const title = slot.dataset.title ?? name;
    const spec = slot.dataset.spec ?? "";
    const path = `illustrations/${name}.svg`;

    const image = new Image();
    image.alt = title;
    image.addEventListener("load", () => {
      const figure = document.createElement("figure");
      figure.className = "exhibit";
      image.style.maxWidth = "100%";
      const caption = document.createElement("figcaption");
      caption.textContent = title;
      figure.append(image, caption);
      slot.replaceChildren(figure);
    });
    image.addEventListener("error", () => {
      const card = document.createElement("div");
      card.className = "placeholder";
      card.innerHTML = "";
      const heading = document.createElement("strong");
      heading.textContent = `Planned figure: ${title}`;
      const description = document.createElement("span");
      description.textContent = spec;
      const drop = document.createElement("span");
      drop.className = "small mono";
      drop.textContent = `drops in at site/public/${path}`;
      card.append(heading, description, drop);
      slot.replaceChildren(card);
    });
    image.src = publicUrl(path);
  }
}
