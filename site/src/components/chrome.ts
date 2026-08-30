// Shared page chrome: active-nav marking.

export function markActiveNav(): void {
  const here = window.location.pathname.replace(/index\.html$/, "");
  for (const link of document.querySelectorAll<HTMLAnchorElement>(".site-nav a")) {
    const target = new URL(link.href).pathname.replace(/index\.html$/, "");
    if (target === here || (target !== "/" && here.startsWith(target))) {
      link.setAttribute("aria-current", "page");
    }
  }
}
