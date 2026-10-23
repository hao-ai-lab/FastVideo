/** One line of a line diff: kept, removed from the old text, or added in the new. */
export interface DiffLine {
	kind: "same" | "removed" | "added";
	text: string;
}

/**
 * Line-by-line diff of two texts (longest common subsequence), in reading
 * order: within a changed run, removed lines come before added ones.
 */
export function diffLines(oldText: string, newText: string): DiffLine[] {
	const a = oldText.split("\n");
	const b = newText.split("\n");
	// lcs[i][j]: length of the longest common subsequence of a[i:] and b[j:]
	const lcs = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
	for (let i = a.length - 1; i >= 0; i--) {
		for (let j = b.length - 1; j >= 0; j--) {
			lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
		}
	}
	const out: DiffLine[] = [];
	let i = 0;
	let j = 0;
	while (i < a.length || j < b.length) {
		if (i < a.length && j < b.length && a[i] === b[j]) {
			out.push({ kind: "same", text: a[i] });
			i++;
			j++;
		} else if (j >= b.length || (i < a.length && lcs[i + 1][j] >= lcs[i][j + 1])) {
			out.push({ kind: "removed", text: a[i] });
			i++;
		} else {
			out.push({ kind: "added", text: b[j] });
			j++;
		}
	}
	return out;
}
