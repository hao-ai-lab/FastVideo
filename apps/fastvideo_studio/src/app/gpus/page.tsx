'use client';

import GpuGrid from '@/components/system/GpuGrid';

export default function GpusPage() {
  return (
    <main className="mx-auto flex w-full max-w-[1100px] flex-col gap-6 px-4 pb-12 pt-6">
      <GpuGrid />
    </main>
  );
}
