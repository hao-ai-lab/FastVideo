'use client';

import { Button } from '@/components/ui/button';
import { setCreateDatasetModalOpen } from '@/stores/createDatasetModalOpen';

export default function AddDatasetButton() {
  return (
    <Button type="button" onClick={() => setCreateDatasetModalOpen(true)}>
      Add Dataset
    </Button>
  );
}
