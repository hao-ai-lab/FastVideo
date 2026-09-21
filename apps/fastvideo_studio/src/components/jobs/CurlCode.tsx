import * as React from 'react';
import { highlightCurl, type CurlTokenKind } from '@/lib/highlightCurl';

const COLORS: Record<CurlTokenKind, string> = {
  plain: '',
  keyword: 'text-violet-700 dark:text-violet-300',
  flag: 'text-sky-700 dark:text-sky-300',
  variable: 'text-amber-800 dark:text-amber-300',
  string: 'text-emerald-700 dark:text-emerald-300',
  key: 'text-blue-700 dark:text-blue-300',
  number: 'text-orange-700 dark:text-orange-300',
  literal: 'text-violet-700 dark:text-violet-300',
  comment: 'text-slate-500 dark:text-slate-400',
};

export default function CurlCode({ command }: { command: string }) {
  return <code>{highlightCurl(command).map((token, index) =>
    <span key={index} className={COLORS[token.kind]} data-syntax={token.kind}>{token.text}</span>,
  )}</code>;
}
