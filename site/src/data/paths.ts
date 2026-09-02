// GitHub project pages serve the site below /<repo-name>/, so every asset
// URL goes through the build-time base instead of being root-relative.

export function publicUrl(path: string): string {
  return `${import.meta.env.BASE_URL}${path.replace(/^\//, "")}`;
}
