import { useEffect } from "react";

export const client = "load";
const base = import.meta.env.BASE_URL.replace(/\/$/, "");

/** Keep the existing family recipe selectors working while their React redesign is developed. */
export default function LegacyCookbook({ html }: { html: string }) {
  const body = html.replace(/(href|src)="\/(?!\/|FastVideo(?:\/|"))/g, `$1="${base}/`);
  useEffect(() => {
    const script = document.createElement("script");
    script.src = `${base}/assets/cookbook.js`;
    document.body.append(script);
    return () => script.remove();
  }, [html]);
  return (
    <>
      <link rel="stylesheet" href={`${base}/assets/cookbook-runbooks.css`} />
      <div className="legacy-cookbook" dangerouslySetInnerHTML={{ __html: body }} />
    </>
  );
}
