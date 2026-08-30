// The three evaluation slices as accessible tabs, each stating what a result
// on that slice may and may not be taken to mean.

const SLICES = [
  {
    id: "clean",
    label: "clean",
    what: "No corruption is scheduled; the policy runs undisturbed.",
    may: "Whether extra supervision harmed nominal behavior (a safety check on the trade-off).",
    maynot: "Anything about recovery; nothing goes wrong in these episodes.",
  },
  {
    id: "matched",
    label: "matched corruption",
    what: "One corruption from the same operator used during recovery collection, at held-out times.",
    may: "Whether the recovery arm learned to handle the corruption family it trained on.",
    maynot: "Generalization; the policy may simply have memorized this operator's signature.",
  },
  {
    id: "unseen",
    label: "unseen corruption",
    what: "One corruption from the other derangement, never used in any training data, at disjoint times.",
    may: "The preregistered primary claim: recovery skill that transfers beyond the trained corruption.",
    maynot: "Robustness to arbitrary failures; 'unseen' here means the one other derangement of a three-action set, not an open world.",
  },
] as const;

export function renderSliceExplainer(mount: HTMLElement): void {
  const wrapper = document.createElement("div");
  wrapper.className = "tabs";
  wrapper.innerHTML = `
    <div role="tablist" aria-label="evaluation slices">
      ${SLICES.map(
        (slice, index) => `
        <button role="tab" id="tab-${slice.id}" aria-controls="panel-${slice.id}"
                aria-selected="${index === 0}" tabindex="${index === 0 ? 0 : -1}">
          ${slice.label}
        </button>`,
      ).join("")}
    </div>
    ${SLICES.map(
      (slice, index) => `
      <div role="tabpanel" id="panel-${slice.id}" aria-labelledby="tab-${slice.id}"
           ${index === 0 ? "" : "hidden"} class="prose">
        <p>${slice.what}</p>
        <p><b>A result here can support:</b> ${slice.may}</p>
        <p><b>It cannot support:</b> ${slice.maynot}</p>
      </div>`,
    ).join("")}
  `;
  const tabs = [...wrapper.querySelectorAll<HTMLButtonElement>("[role=tab]")];
  const panels = [...wrapper.querySelectorAll<HTMLElement>("[role=tabpanel]")];
  const select = (index: number) => {
    tabs.forEach((tab, i) => {
      tab.setAttribute("aria-selected", String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      panels[i]!.hidden = i !== index;
    });
    tabs[index]!.focus();
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(index));
    tab.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight") select((index + 1) % tabs.length);
      if (event.key === "ArrowLeft") select((index + tabs.length - 1) % tabs.length);
    });
  });
  mount.replaceChildren(wrapper);
}
