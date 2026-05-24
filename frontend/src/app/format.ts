import { marked } from 'marked';
import katex from 'katex';
import DOMPurify from 'dompurify';

/**
 * Render assistant text as safe HTML: LaTeX math ($…$ / $$…$$) via KaTeX, then
 * Markdown via marked, then sanitised with DOMPurify.
 *
 * Math is extracted to placeholders BEFORE Markdown so marked can't mangle it
 * (e.g. treat `x_1` underscores as emphasis), then re-inserted afterwards.
 */
export function renderRich(text: string): string {
  if (!text) return '';

  const math: string[] = [];
  const protect = (expr: string, display: boolean): string => {
    let html: string;
    try {
      html = katex.renderToString(expr.trim(), { displayMode: display, throwOnError: false, output: 'html' });
    } catch {
      html = (display ? `$$${expr}$$` : `$${expr}$`);
    }
    math.push(html);
    return `@@MATH${math.length - 1}@@`;
  };

  let src = text.replace(/\$\$([\s\S]+?)\$\$/g, (_m, e) => protect(e, true));
  src = src.replace(/\$([^$\n]+?)\$/g, (_m, e) => protect(e, false));

  let html = marked.parse(src, { async: false }) as string;
  html = html.replace(/@@MATH(\d+)@@/g, (_m, i) => math[+i] ?? '');

  return DOMPurify.sanitize(html);
}
