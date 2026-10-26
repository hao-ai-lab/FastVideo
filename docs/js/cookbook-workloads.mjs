/**
 * Shared workload IDs and media direction. Endpoint and request-field choices mirror the
 * serving API contract; callers must not infer support from an ID alone.
 */
export const WORKLOADS = Object.freeze({
  t2v: Object.freeze({
    input: "text",
    output: "video",
    endpoint: "/v1/videos/sync",
    extension: "mp4",
    media_type: "video/mp4",
  }),
  i2v: Object.freeze({
    input: "image",
    output: "video",
    endpoint: "/v1/videos/sync",
    extension: "mp4",
    media_type: "video/mp4",
    reference: "input_reference",
  }),
  t2i: Object.freeze({
    input: "text",
    output: "image",
    endpoint: "/v1/images/generations",
    extension: "json",
    media_type: "application/json",
  }),
  t2a: Object.freeze({
    input: "text",
    output: "audio",
    endpoint: "/v1/videos/sync",
    extension: "wav",
    media_type: "audio/wav",
  }),
  v2a: Object.freeze({
    input: "video",
    output: "audio",
    endpoint: "/v1/videos/sync",
    extension: "wav",
    media_type: "audio/wav",
    reference: "video_reference",
  }),
});

export const WORKLOAD_IDS = Object.freeze(Object.keys(WORKLOADS));
