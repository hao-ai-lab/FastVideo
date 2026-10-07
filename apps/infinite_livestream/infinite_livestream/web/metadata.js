// hls.js and native Safari expose timed ID3 as metadata TextTrack cues.
// Reuse those cues; where available, select by the actual presented frame PTS.
(function (scope) {
  "use strict";

  function recordFromCue(cue) {
    try {
      const value = cue.value || JSON.parse(cue.text || "null");
      if (!value || value.key !== "TXXX" || typeof value.data !== "string") return undefined;
      if (value.info && value.info !== "infinite-livestream") return undefined;
      const record = JSON.parse(value.data);
      if (record.version !== 1) return undefined;
      if (record.clip === null) return record;
      const clip = record.clip;
      if (!clip || typeof clip.clip_id !== "string" || typeof clip.title !== "string" ||
          typeof clip.prompt !== "string" || typeof clip.generated !== "boolean") return undefined;
      return record;
    } catch (_) {
      return undefined;
    }
  }

  function watch(video, onChange) {
    const tracks = new Map();
    let previous;
    let disposed = false;
    const frameClock = typeof video.requestVideoFrameCallback === "function";
    let frameTime;
    let frameCallback;

    function presented(_now, metadata) {
      frameTime = Number.isFinite(metadata.mediaTime) ? metadata.mediaTime : undefined;
      update();
      if (!disposed) frameCallback = video.requestVideoFrameCallback(presented);
    }

    function update() {
      if (disposed) return;
      let selected;
      let latest = -Infinity;
      if (video.readyState >= 2) {
        // Firefox can display a segment's first frame after a paused seek while
        // currentTime is still just before its PTS. activeCues then names the
        // old clip. Presented-frame time resolves this without adjusting clocks.
        const hasFrame = frameTime !== undefined;
        const at = Math.round(frameTime * 1e6);
        for (const track of tracks.keys()) {
          const cues = (hasFrame ? track.cues : track.activeCues) || [];
          let end = cues.length;
          if (hasFrame) {
            // TextTrackCueList is sorted by start time. Binary search avoids
            // scanning a long viewer's entire buffered history every frame.
            let lo = 0;
            while (lo < end) {
              const mid = (lo + end) >>> 1;
              // Browsers report frame timestamps at microsecond resolution.
              if (Math.round(cues[mid].startTime * 1e6) <= at) lo = mid + 1;
              else end = mid;
            }
          }
          for (let i = end - 1; i >= 0; i--) {
            const cue = cues[i];
            if (hasFrame && Math.round(cue.endTime * 1e6) <= at) continue;
            const record = recordFromCue(cue);
            // Native players can leave older ID3 cues open. Prefer the latest
            // applicable record, including the explicit empty/black record.
            if (record && cue.startTime >= latest) {
              latest = cue.startTime;
              selected = record.clip;
            }
            if (hasFrame && record) break;
          }
        }
      }
      const key = selected === undefined ? undefined : JSON.stringify(selected);
      if (key !== previous) {
        previous = key;
        onChange(selected);
      }
    }

    function add(track) {
      if (track.kind !== "metadata" || tracks.has(track)) return;
      track.mode = "hidden";
      track.addEventListener("cuechange", update);
      tracks.set(track, update);
      update();
    }
    function added(event) { add(event.track); }
    function removed(event) {
      event.track.removeEventListener("cuechange", update);
      tracks.delete(event.track);
      update();
    }
    function seeked() {
      // Some browsers do not issue a frame callback for a paused seek. Their
      // cue clock remains the fallback until another presented frame arrives.
      frameTime = undefined;
      update();
    }
    function emptied() {
      frameTime = undefined;
      previous = undefined;
      onChange(undefined);
    }

    video.textTracks.addEventListener("addtrack", added);
    video.textTracks.addEventListener("removetrack", removed);
    video.addEventListener("loadeddata", update);
    video.addEventListener("seeked", seeked);
    video.addEventListener("playing", update);
    video.addEventListener("emptied", emptied);
    onChange(undefined);
    for (const track of Array.from(video.textTracks)) add(track);
    if (frameClock) frameCallback = video.requestVideoFrameCallback(presented);

    const stop = () => {
      disposed = true;
      if (frameClock) video.cancelVideoFrameCallback(frameCallback);
      video.textTracks.removeEventListener("addtrack", added);
      video.textTracks.removeEventListener("removetrack", removed);
      video.removeEventListener("loadeddata", update);
      video.removeEventListener("seeked", seeked);
      video.removeEventListener("playing", update);
      video.removeEventListener("emptied", emptied);
      for (const track of tracks.keys()) track.removeEventListener("cuechange", update);
      tracks.clear();
    };
    stop.inspect = () => ({
      frameTime,
      tracks: Array.from(tracks.keys(), track => ({ count: track.cues?.length || 0, start: track.cues?.[0]?.startTime })),
    });
    return stop;
  }

  scope.LivestreamMetadata = { watch };
})(globalThis);
