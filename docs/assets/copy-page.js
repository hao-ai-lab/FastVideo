// Add a one-click copy button for the rendered MkDocs article.
(function () {
  const BUTTON_ID = "copy-page-button";

  function text(value) {
    return (value || "").replace(/\s+/g, " ").trim();
  }

  function codeLanguage(code) {
    const classes = Array.from(code.classList || []);
    const language = classes.find((name) => name.startsWith("language-"));
    return language ? language.replace("language-", "") : "";
  }

  function serializeInline(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.textContent || "";
    }

    if (node.nodeType !== Node.ELEMENT_NODE) {
      return "";
    }

    const tagName = node.tagName.toLowerCase();

    if (tagName === "code" && node.parentElement && node.parentElement.tagName.toLowerCase() !== "pre") {
      return "`" + (node.textContent || "").trim() + "`";
    }

    if (tagName === "a") {
      if (node.classList.contains("headerlink")) {
        return "";
      }

      const label = text(Array.from(node.childNodes).map(serializeInline).join(""));
      const href = node.href;
      return href && label ? `${label} (${href})` : label;
    }

    if (tagName === "img") {
      const alt = node.getAttribute("alt") || "image";
      const src = node.src || "";
      return src ? `[${alt}](${src})` : `[${alt}]`;
    }

    if (tagName === "br") {
      return "\n";
    }

    return Array.from(node.childNodes).map(serializeInline).join("");
  }

  function serializeTable(table) {
    const rows = Array.from(table.rows);
    if (rows.length === 0) return "";

    const mdRows = rows.map(
      (row) => "| " + Array.from(row.children).map((cell) => text(serializeInline(cell))).join(" | ") + " |"
    );
    const separator = "| " + Array.from(rows[0].children)
      .map(() => "---")
      .join(" | ") + " |";
    mdRows.splice(1, 0, separator);
    return mdRows.join("\n");
  }

  function serializeBlock(node, listDepth = 0) {
    if (node.nodeType === Node.TEXT_NODE) {
      return text(node.textContent);
    }

    if (node.nodeType !== Node.ELEMENT_NODE) {
      return "";
    }

    const tagName = node.tagName.toLowerCase();

    if (["script", "style", "nav", "button"].includes(tagName) || node.id === BUTTON_ID) {
      return "";
    }

    if (/^h[1-6]$/.test(tagName)) {
      const level = Number(tagName.slice(1));
      return `${"#".repeat(level)} ${text(serializeInline(node))}`;
    }

    if (tagName === "pre") {
      const code = node.querySelector("code");
      const content = code ? code.textContent || "" : node.textContent || "";
      return `\`\`\`${code ? codeLanguage(code) : ""}\n${content.replace(/\n$/, "")}\n\`\`\``;
    }

    if (["p", "figcaption"].includes(tagName)) {
      return text(serializeInline(node));
    }

    if (tagName === "blockquote") {
      return serializeChildren(node, listDepth)
        .split("\n")
        .map((line) => (line ? `> ${line}` : ">"))
        .join("\n");
    }

    if (tagName === "ul" || tagName === "ol") {
      return Array.from(node.children)
        .filter((child) => child.tagName && child.tagName.toLowerCase() === "li")
        .map((item, index) => serializeListItem(item, tagName === "ol", index, listDepth))
        .join("\n");
    }

    if (tagName === "table") {
      return serializeTable(node);
    }

    if (["hr"].includes(tagName)) {
      return "---";
    }

    return serializeChildren(node, listDepth);
  }

  function serializeListItem(item, ordered, index, listDepth) {
    const marker = ordered ? `${index + 1}. ` : "- ";
    const indent = "  ".repeat(listDepth);
    const childBlocks = [];
    const inlineParts = [];

    Array.from(item.childNodes).forEach((child) => {
      if (child.nodeType === Node.ELEMENT_NODE && ["ul", "ol"].includes(child.tagName.toLowerCase())) {
        childBlocks.push(serializeBlock(child, listDepth + 1));
      } else {
        const content = serializeInline(child);
        if (content) inlineParts.push(content);
      }
    });

    const firstLine = `${indent}${marker}${text(inlineParts.join(" "))}`.trimEnd();
    return [firstLine, ...childBlocks.filter(Boolean)].join("\n");
  }

  function serializeChildren(node, listDepth = 0) {
    return Array.from(node.childNodes)
      .map((child) => serializeBlock(child, listDepth))
      .map((value) => value.trim())
      .filter(Boolean)
      .join("\n\n");
  }

  function articleText(article) {
    const clone = article.cloneNode(true);
    clone.querySelectorAll("script, style, .headerlink, .md-clipboard, #copy-page-button").forEach((node) => node.remove());

    const content = serializeChildren(clone).trim();
    const title = document.querySelector("h1") || document.querySelector("title");
    const pageTitle = title ? text(title.textContent) : "";

    if (pageTitle && !content.startsWith("# ")) {
      return `# ${pageTitle}\n\n${content}`.trim();
    }

    return content;
  }

  async function copyArticle(button, article) {
    const originalLabel = button.textContent;
    try {
      await navigator.clipboard.writeText(articleText(article));
      button.textContent = "Copied!";
      button.classList.add("copy-page-button--copied");
    } catch (error) {
      button.textContent = "Copy failed";
      button.classList.add("copy-page-button--error");
      console.error("Failed to copy page", error);
    }

    window.setTimeout(() => {
      button.textContent = originalLabel;
      button.classList.remove("copy-page-button--copied", "copy-page-button--error");
    }, 2000);
  }

  function addCopyButton() {
    const article = document.querySelector("article.md-content__inner");
    if (!article || document.getElementById(BUTTON_ID)) {
      return;
    }

    const button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.className = "copy-page-button md-button md-button--primary";
    button.textContent = "Copy page";
    button.setAttribute("aria-label", "Copy this page as plain text");
    button.addEventListener("click", () => copyArticle(button, article));

    article.insertBefore(button, article.firstChild);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(addCopyButton);
  }

  document.addEventListener("DOMContentLoaded", addCopyButton);
  window.addEventListener("load", addCopyButton);
})();
