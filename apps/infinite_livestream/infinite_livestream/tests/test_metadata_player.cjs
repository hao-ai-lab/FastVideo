// Adapter regressions; actual cue timing is owned by the browser/player.
const { test } = require('node:test');
const assert = require('node:assert/strict');
require('../web/metadata.js');

class Track extends EventTarget {
  kind = 'metadata';
  mode = 'disabled';
  activeCues = [];
  activate(cues) {
    this.activeCues = cues;
    this.dispatchEvent(new Event('cuechange'));
  }
}
class TrackList extends EventTarget {
  items = [];
  [Symbol.iterator]() { return this.items[Symbol.iterator](); }
  add(track) {
    if (!this.items.includes(track)) this.items.push(track);
    const event = new Event('addtrack');
    event.track = track;
    this.dispatchEvent(event);
  }
}
class Video extends EventTarget {
  readyState = 2;
  textTracks = new TrackList();
}
function cue(name, startTime = 0, native = false) {
  const clip = name === null ? null : { clip_id: name, title: name, prompt: name, generated: false };
  const value = { key: 'TXXX', data: JSON.stringify({ version: 1, clip }) };
  if (!native) value.info = 'infinite-livestream';
  return { startTime, endTime: Infinity, value };
}
function player() {
  const video = new Video();
  const track = new Track();
  video.textTracks.add(track);
  const shown = [];
  const stop = LivestreamMetadata.watch(video, clip => shown.push(clip?.clip_id ?? clip));
  return { video, track, shown, stop };
}

test('downloaded future cues do not advance the title; activation does', () => {
  const p = player();
  assert.equal(p.track.mode, 'hidden');
  const a = cue('A');
  const b = cue('B', 10);
  p.track.activate([a]);
  p.track.cues = [a, b]; // The next segment arrived, but it is not playing.
  p.video.dispatchEvent(new Event('loadeddata'));
  assert.deepEqual(p.shown, [undefined, 'A']);
  p.track.activate([b]);
  assert.deepEqual(p.shown, [undefined, 'A', 'B']);
  p.stop();
});

test('viewers keep independent titles during stalls and seeks', () => {
  const live = player();
  const delayed = player();
  live.track.activate([cue('C', 20)]);
  delayed.track.activate([cue('A', 0)]);
  delayed.video.dispatchEvent(new Event('waiting'));
  live.track.activate([cue('D', 30)]);
  assert.equal(delayed.shown.at(-1), 'A');
  delayed.track.activate([cue('B', 10)]);
  delayed.video.dispatchEvent(new Event('seeked'));
  assert.equal(delayed.shown.at(-1), 'B');
  assert.equal(live.shown.at(-1), 'D');
  delayed.track.activate([cue('A')]); // Seek backwards.
  assert.equal(delayed.shown.at(-1), 'A');
  live.stop(); delayed.stop();
});

test('joining mid-clip and repeated segment markers do not need prior events', () => {
  const video = new Video();
  const shown = [];
  const stop = LivestreamMetadata.watch(video, clip => shown.push(clip?.clip_id ?? clip));
  const track = new Track();
  track.activeCues = [cue('B', 14)];
  video.textTracks.add(track);
  track.activate([cue('B', 16)]);
  video.textTracks.add(track); // hls.js may reuse a track across attachments.
  assert.deepEqual(shown, [undefined, 'B']);
  stop();
});

test('native overlapping cues pick the latest record, including black frames', () => {
  const p = player();
  p.track.activate([cue('B', 10, true), cue('A', 0, true)]);
  assert.equal(p.shown.at(-1), 'B');
  p.track.activate([cue('A', 0, true), cue(null, 20, true), cue('B', 10, true)]);
  assert.equal(p.shown.at(-1), null);
  p.stop();
});

test('invalid and unrelated ID3 cannot replace a valid title', () => {
  const p = player();
  const good = cue('猫 <script> & café', 1);
  const unrelated = cue('wrong', 10);
  unrelated.value.info = 'another-application';
  p.track.activate([good, unrelated, { startTime: 20, value: { key: 'TXXX', data: 'invalid' } }]);
  assert.equal(p.shown.at(-1), '猫 <script> & café');
  const fallback = cue('text-fallback', 30);
  fallback.text = JSON.stringify(fallback.value);
  delete fallback.value;
  p.track.activate([fallback]);
  assert.equal(p.shown.at(-1), 'text-fallback');
  p.stop();
});

test('loading and teardown clear stale titles and detach old listeners', () => {
  const p = player();
  p.track.activate([cue('A')]);
  p.video.readyState = 0;
  p.video.dispatchEvent(new Event('emptied'));
  p.track.activate([cue('B')]);
  assert.equal(p.shown.at(-1), undefined);
  p.video.readyState = 2;
  p.video.dispatchEvent(new Event('loadeddata'));
  assert.equal(p.shown.at(-1), 'B');
  p.stop();
  p.track.activate([cue('C')]);
  p.video.textTracks.add(new Track());
  p.video.dispatchEvent(new Event('emptied'));
  assert.equal(p.shown.at(-1), 'B');
});


test('presented frames resolve paused seeks before segment PTS without guessing', () => {
  const video = new Video();
  let callback;
  let cancelled = false;
  video.requestVideoFrameCallback = fn => { callback = fn; return 1; };
  video.cancelVideoFrameCallback = id => { assert.equal(id, 1); cancelled = true; };
  const track = new Track();
  video.textTracks.add(track);
  const shown = [];
  const stop = LivestreamMetadata.watch(video, clip => shown.push(clip?.clip_id ?? clip));
  const a = cue('A', 2.021333333333333);
  const b = cue('B', 12.021333333333333);
  a.endTime = b.startTime;
  track.cues = [a, b];
  track.activate([a]);
  callback(0, { mediaTime: 3 });
  assert.equal(shown.at(-1), 'A');
  // Observed Firefox case: currentTime=12, active cue A, but displayed frame B.
  video.currentTime = 12;
  callback(0, { mediaTime: 12.021333 });
  assert.equal(shown.at(-1), 'B');
  track.activate([a]);
  assert.equal(shown.at(-1), 'B');
  // A future download and a stall do not move the presented frame.
  track.cues.push(cue('C', 20));
  video.dispatchEvent(new Event('waiting'));
  assert.equal(shown.at(-1), 'B');
  callback(0, { mediaTime: 3 });
  assert.equal(shown.at(-1), 'A');
  // A paused seek may have no frame callback: resume standard cue scheduling.
  track.activate([b]);
  video.dispatchEvent(new Event('seeked'));
  assert.equal(shown.at(-1), 'B');
  video.dispatchEvent(new Event('emptied'));
  assert.equal(shown.at(-1), undefined);
  stop();
  assert.equal(cancelled, true);
});


test('a browser advertising native HLS still uses the metadata-capable hls.js path', () => {
  const fs = require('node:fs');
  const vm = require('node:vm');
  const html = fs.readFileSync(require.resolve('../web/index.html'), 'utf8');
  const source = html.slice(html.indexOf('function startPlayback()'), html.indexOf('video.addEventListener("error"'));
  let loaded;
  let attached;
  class Hls {
    static isSupported() { return true; }
    static Events = { ERROR: 'error' };
    on() {}
    loadSource(url) { loaded = url; }
    attachMedia(video) { attached = video; }
  }
  const video = { canPlayType: () => 'probably' };
  const context = vm.createContext({
    Hls, window: { Hls }, video, PLAYLIST: '/hls/stream.m3u8',
    LivestreamMetadata: { watch: () => () => {} }, renderPlayback() {},
  });
  vm.runInContext(source + '\nstartPlayback();', context);
  assert.equal(loaded, '/hls/stream.m3u8');
  assert.equal(attached, video);
  assert.equal(video.src, undefined);
  // Preserve native-only devices when Media Source playback is unavailable.
  Hls.isSupported = () => false;
  loaded = undefined;
  vm.runInContext('startPlayback();', context);
  assert.equal(video.src, '/hls/stream.m3u8');
  assert.equal(loaded, undefined);
});

test('missing metadata does not claim an already-playing video is loading', () => {
  const fs = require('node:fs');
  const vm = require('node:vm');
  const html = fs.readFileSync(require.resolve('../web/index.html'), 'utf8');
  const source = html.slice(html.indexOf('function renderPlayback()'), html.indexOf('function render(state)'));
  const elements = {};
  const video = { readyState: 4, paused: false };
  const context = vm.createContext({
    video, playbackClip: undefined, playbackError: '', buffering: false,
    el: id => elements[id] ||= {},
  });
  vm.runInContext(source + '\nrenderPlayback();', context);
  assert.equal(elements.livetext.textContent, 'on air');
  assert.equal(elements['np-title'].textContent, 'Waiting for title metadata');
  video.readyState = 0;
  vm.runInContext('renderPlayback();', context);
  assert.equal(elements.livetext.textContent, 'loading');
  assert.equal(elements['np-title'].textContent, 'Waiting for video');
});
