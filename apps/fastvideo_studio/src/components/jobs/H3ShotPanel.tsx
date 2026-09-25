'use client';

import * as React from 'react';

import {
  CAMERA_MOVEMENTS,
  SHOT_TYPES,
  shotLines,
  shotProse,
  type CameraMovement,
  type Shot,
  type ShotType,
} from '@/lib/h3Shots';

const TYPE_LABEL = Object.fromEntries(SHOT_TYPES.map((t) => [t.id, t.label])) as Record<ShotType, string>;
const MOVEMENT_LABEL = Object.fromEntries(CAMERA_MOVEMENTS.map((m) => [m.id, m.label])) as Record<
  CameraMovement,
  string
>;

// Schematic framing per shot type: `scale` sizes the figures and `baseY` is
// where their shoulders bottom out in the 160x90 frame, so a wide shot reads
// as small distant figures and a close-up as a head filling the frame.
const FRAMING: Partial<Record<ShotType, { scale: number; baseY: number }>> = {
  establishing: { scale: 0.45, baseY: 72 },
  wide: { scale: 0.8, baseY: 78 },
  medium: { scale: 1.5, baseY: 92 },
  'close-up': { scale: 2.6, baseY: 118 },
};

function figureXs(count: number): number[] {
  if (count <= 1) return [80];
  if (count === 2) return [55, 105];
  return [38, 80, 122];
}

function Figure({ x, baseY, scale, opacity }: { x: number; baseY: number; scale: number; opacity: number }) {
  return (
    <g fill="currentColor" opacity={opacity}>
      <circle cx={x} cy={baseY - 30 * scale} r={8 * scale} />
      <ellipse cx={x} cy={baseY - 12 * scale} rx={14 * scale} ry={12 * scale} />
    </g>
  );
}

export function ShotFrame({ shot }: { shot: Shot }) {
  const count = Math.min(Math.max(shot.subjectIds.length, 1), 3);
  const empty = shot.subjectIds.length === 0;
  const opacity = empty ? 0.25 : 0.6;
  const framing = FRAMING[shot.type];

  let scene: React.ReactNode;
  if (shot.type === 'extreme-close-up') {
    scene = (
      <g fill="currentColor" opacity={opacity}>
        <circle cx={80} cy={52} r={62} />
        <ellipse cx={60} cy={42} rx={7} ry={4} className="fill-background" />
        <ellipse cx={100} cy={42} rx={7} ry={4} className="fill-background" />
      </g>
    );
  } else if (shot.type === 'over-the-shoulder') {
    scene = (
      <>
        <Figure x={22} baseY={135} scale={3} opacity={0.85} />
        <Figure x={116} baseY={80} scale={1} opacity={opacity} />
      </>
    );
  } else if (shot.type === 'pov') {
    scene = (
      <>
        <Figure x={80} baseY={72} scale={0.9} opacity={opacity} />
        <path
          d="M0 0H160V90H0Z M80 45m-52 0a52 40 0 1 0 104 0a52 40 0 1 0 -104 0"
          fill="currentColor"
          fillRule="evenodd"
          opacity={0.3}
        />
        <path d="M74 45H86M80 39V51" stroke="currentColor" strokeWidth={1.5} opacity={0.6} />
      </>
    );
  } else if (shot.type === 'insert') {
    scene = (
      <g opacity={0.6} fill="none" stroke="currentColor" strokeWidth={2.5}>
        <rect x={52} y={16} width={56} height={58} rx={8} />
        <path d="M64 34H96M64 46H96M64 58H84" strokeLinecap="round" />
      </g>
    );
  } else if (framing) {
    scene = (
      <>
        {shot.type === 'establishing' && (
          <path d="M0 66H160" stroke="currentColor" strokeWidth={1} opacity={0.35} />
        )}
        {figureXs(count).map((x, i) => (
          <Figure
            key={x}
            x={shot.type === 'close-up' ? 80 : x}
            baseY={framing.baseY}
            scale={framing.scale}
            opacity={opacity * (i === 0 ? 1 : 0.75)}
          />
        ))}
      </>
    );
  }

  return (
    <svg viewBox="0 0 160 90" className="block h-full w-full text-foreground" aria-hidden>
      {scene}
      <MovementOverlay movement={shot.movement} />
    </svg>
  );
}

function MovementOverlay({ movement }: { movement: CameraMovement }) {
  const g = (children: React.ReactNode) => (
    <g
      className="text-accent-blue"
      stroke="currentColor"
      fill="none"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {children}
    </g>
  );
  switch (movement) {
    case 'pan':
      return g(<path d="M28 80H132M28 80l6-4M28 80l6 4M132 80l-6-4M132 80l-6 4" />);
    case 'tilt':
      return g(<path d="M146 18V72M146 18l-4 6M146 18l4 6M146 72l-4-6M146 72l4-6" />);
    case 'dolly':
      return g(<path d="M10 86L54 56M150 86L106 56M10 86H150" strokeDasharray="4 4" />);
    case 'zoom':
      return g(<path d="M16 26V14H28M132 14H144V26M16 64V76H28M132 76H144V64" />);
    case 'handheld':
      return g(
        <rect x={7} y={7} width={146} height={76} rx={4} strokeDasharray="3 4" transform="rotate(-1.6 80 45)" />,
      );
    case 'tracking':
      return g(
        <>
          <path d="M22 82H134M134 82l-6-4M134 82l-6 4" />
          <circle cx={30} cy={82} r={3} fill="currentColor" />
        </>,
      );
    case 'crane':
      return g(<path d="M24 80L134 20M134 20h-9M134 20l-3 8" />);
    default:
      return null;
  }
}

function subjectShortLabel(label: string): string {
  const m = label.match(/(\d+)/);
  return m ? `Subj ${m[1]}` : label;
}

export interface H3ShotPanelProps {
  shot: Shot;
  /** the number shown in the badge */
  number: number | string;
  /** time printed on the frame, e.g. "00:03.500" */
  timeLabel: string;
}

/**
 * The contents of one storyboard panel: badge, shot type and camera move, a
 * schematic frame, who is in it, what is said, and the shot's prose. Callers
 * supply the surrounding card (a button in the editor, plain in the scene view).
 */
export function H3ShotPanel({ shot, number, timeLabel }: H3ShotPanelProps) {
  const spoken = shotLines(shot).filter((l) => l.text.trim());
  return (
    <>
      <div className="flex items-center gap-1.5">
        <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-secondary text-[11px] font-semibold text-foreground">
          {number}
        </span>
        <span className="truncate text-xs font-medium text-foreground">{TYPE_LABEL[shot.type]}</span>
        <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">{MOVEMENT_LABEL[shot.movement]}</span>
      </div>

      <div className="relative aspect-video overflow-hidden rounded-md border border-border bg-secondary/40">
        <ShotFrame shot={shot} />
        <span className="absolute bottom-1 left-1 rounded bg-background/80 px-1 font-mono text-[9px] text-muted-foreground">
          {timeLabel}
        </span>
      </div>

      {shot.subjectIds.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {shot.subjectIds.map((label) => (
            <span
              key={label}
              className="rounded-full border border-input bg-secondary/50 px-1.5 py-px font-mono text-[10px] text-muted-foreground"
            >
              {subjectShortLabel(label)}
            </span>
          ))}
        </div>
      )}

      {spoken.length > 0 && (
        <p className="line-clamp-3 rounded-md bg-accent-blue/10 px-1.5 py-1 text-[11px] italic text-foreground">
          {spoken.map((l) => `${l.speaker.trim() ? `${l.speaker.trim()}: ` : ''}“${l.text.trim()}”`).join(' ')}
        </p>
      )}

      <p className="line-clamp-2 text-[11px] text-muted-foreground">{shotProse(shot) || 'No description yet'}</p>
    </>
  );
}
