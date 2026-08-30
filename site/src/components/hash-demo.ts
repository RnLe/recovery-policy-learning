// Why a frozen protocol is mechanical rather than a promise: edit a field and
// watch the fingerprint stop matching. The digest itself is deliberately not
// shown. A page of hexadecimal proves nothing to a reader; what the mechanism
// actually guarantees is that *any* edit is detectable, and that is what this
// reports.

const FIELDS = [
  { key: "n0", label: "base labels (n0)", value: "4000" },
  { key: "b", label: "budget per round (b)", value: "1000" },
  { key: "h", label: "recovery window (h)", value: "8" },
  { key: "sesoi", label: "SESOI", value: "0.05" },
] as const;

function canonical(document: Record<string, string>): string {
  const keys = Object.keys(document).sort();
  return JSON.stringify(
    Object.fromEntries(keys.map((key) => [key, document[key]])),
  );
}

async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function renderHashDemo(mount: HTMLElement): void {
  const wrapper = document.createElement("div");
  wrapper.className = "hash-demo";
  wrapper.innerHTML = `
    <p class="small">A live illustration of the mechanism: edit any
      field.</p>
    <div class="fields">
      ${FIELDS.map(
        (field) => `
        <label>${field.label}
          <input name="${field.key}" value="${field.value}" autocomplete="off"
                 spellcheck="false">
        </label>`,
      ).join("")}
    </div>
    <output aria-live="polite"></output>
    <p class="small muted">The fingerprint is a SHA-256 digest of the canonical
      JSON above. Changing one character anywhere changes all of it, so a quiet
      edit to a frozen protocol cannot survive comparison against the recorded
      fingerprint.</p>
  `;
  const output = wrapper.querySelector<HTMLOutputElement>("output")!;
  const inputs = [...wrapper.querySelectorAll<HTMLInputElement>("input")];
  let initial: string | null = null;

  const update = async () => {
    const doc = Object.fromEntries(inputs.map((i) => [i.name, i.value]));
    try {
      const hash = await sha256Hex(canonical(doc));
      initial ??= hash;
      const changed = hash !== initial;
      output.textContent = changed
        ? "fingerprint no longer matches the frozen contract"
        : "fingerprint matches the frozen contract";
      output.classList.toggle("changed", changed);
    } catch {
      output.textContent = "hashing unavailable in this browser context";
    }
  };
  for (const input of inputs) {
    input.addEventListener("input", () => void update());
  }
  void update();
  mount.replaceChildren(wrapper);
}
