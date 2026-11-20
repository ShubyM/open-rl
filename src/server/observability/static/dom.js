// Keyed DOM morphing so periodic refreshes keep focus, open popovers, scroll position, and hover state.

const keyOf = (node) => node.nodeType === 1 && (node.id || node.getAttribute("data-key"));

export function morph(container, html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  morphChildren(container, template.content);
}

function morphChildren(from, to) {
  const keyed = new Map(Array.from(from.childNodes).filter(keyOf).map((node) => [keyOf(node), node]));
  let cursor = from.firstChild;
  for (const next of Array.from(to.childNodes)) {
    const key = keyOf(next);
    let node = key ? keyed.get(key) : cursor && !keyOf(cursor) ? cursor : null;
    if (!node) from.insertBefore((node = next), cursor);
    else {
      if (node !== cursor) from.insertBefore(node, cursor);
      node = morphNode(node, next);
    }
    cursor = node.nextSibling;
  }
  while (cursor) {
    const next = cursor.nextSibling;
    cursor.remove();
    cursor = next;
  }
}

function morphNode(node, next) {
  if (node.isEqualNode(next)) return node;
  if (node.nodeType !== next.nodeType || (node.nodeType === 1 && (node.tagName !== next.tagName || node.namespaceURI !== next.namespaceURI))) {
    node.replaceWith(next);
    return next;
  }
  if (node.nodeType !== 1) {
    if (node.data !== next.data) node.data = next.data;
    return node;
  }
  if (node.classList.contains("chart-hover") || node.classList.contains("activity-tooltip")) return node;
  if (node.classList.contains("chart-plot") && ["chartId", "start", "end", "min", "max"].some((k) => node.dataset[k] !== next.dataset[k])) {
    const hover = node.querySelector(".chart-hover");
    if (hover) hover.hidden = true;
  }
  const preserve = (name) => (name === "open" && node.tagName === "DETAILS") || (name === "data-crosshair" || name === "style" && node.hasAttribute("data-crosshair-host"));
  const nextValue = next.value, nextChecked = next.checked;
  for (const { name } of Array.from(node.attributes)) if (!preserve(name) && !next.hasAttribute(name)) node.removeAttribute(name);
  for (const { name, value } of Array.from(next.attributes)) if (!preserve(name) && node.getAttribute(name) !== value) node.setAttribute(name, value);
  morphChildren(node, next);
  if ((node.tagName === "SELECT" || (node.tagName === "INPUT" && node.type !== "checkbox" && node !== document.activeElement)) && node.value !== nextValue) {
    node.value = nextValue;
  }
  if (node.tagName === "INPUT" && node.type === "checkbox" && node.checked !== nextChecked) node.checked = nextChecked;
  return node;
}

export async function copyLink(url, success = "Link copied") {
  try {
    await navigator.clipboard.writeText(url);
    return success;
  } catch {
    const focused = document.activeElement, input = Object.assign(document.createElement("textarea"), { value: url });
    input.style.cssText = "position:fixed;opacity:0";
    document.body.append(input);
    input.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } catch {}
    input.remove();
    focused?.focus({ preventScroll: true });
    return copied ? success : "Copy failed";
  }
}
