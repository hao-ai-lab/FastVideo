import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"

const load = async (name) => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url), "utf8"))
const [quickstart, tuning] = await Promise.all([load("quickstart.json"), load("tuning.json")])

assert.match(tuning.source.fastvideoCommit, /^[0-9a-f]{40}$/)

const backendIds = new Set(tuning.attentionBackends.map(({ id }) => id))
const workloadIds = new Set(tuning.workloads.map(({ id }) => id))
const modelIds = new Set()
for (const model of tuning.models) {
  assert(!modelIds.has(model.id), `duplicate model: ${model.id}`)
  modelIds.add(model.id)
  assert(workloadIds.has(model.workload), `${model.id}: unknown workload ${model.workload}`)

  const { recipe } = model
  for (const key of ["height", "width", "numFrames", "fps", "numInferenceSteps"]) {
    assert(Number.isInteger(recipe[key]) && recipe[key] > 0, `${model.id}: invalid ${key}`)
  }
  if (model.workload === "game") {
    assert(recipe.numFrames % 12 === 9, `${model.id}: Matrix frames must be 9 + 12k`)
  }
  assert(recipe.guidanceScale > 0, `${model.id}: guidance must be positive`)
  assert(recipe.vsaSparsity >= 0 && recipe.vsaSparsity <= 1, `${model.id}: invalid VSA sparsity`)
  if (recipe.dmdDenoisingSteps !== null) {
    assert(recipe.dmdDenoisingSteps.length > 0 &&
      recipe.dmdDenoisingSteps.every((step) => Number.isInteger(step) && step > 0),
      `${model.id}: invalid DMD schedule`)
  }
  assert(recipe.attentionBackends.length > 0, `${model.id}: no attention backend`)
  assert(recipe.attentionBackends.includes(recipe.defaultAttentionBackend), `${model.id}: invalid default backend`)
  for (const backend of recipe.attentionBackends) {
    assert(backendIds.has(backend), `${model.id}: unknown backend ${backend}`)
  }
}

assert(modelIds.has(tuning.defaults.model_id), "unknown default model")
const profileKeys = new Set()
for (const profile of quickstart.profiles) {
  assert(modelIds.has(profile.model), `quick-start profile uses unknown model: ${profile.model}`)
  assert(tuning.models.find(({ id }) => id === profile.model).workload === profile.task,
    `${profile.model}: quick-start task does not match recipe workload`)
  const key = `${profile.task}:${profile.tier}`
  assert(!profileKeys.has(key), `duplicate quick-start profile: ${key}`)
  profileKeys.add(key)
}

const unsupportedClaims = /"(?:generationSeconds|runtimeSeconds|timeSec|firstRunSec|vramMin|ramMin)"/
assert(!unsupportedClaims.test(JSON.stringify({ quickstart, tuning })), "unproven performance claim in recipe data")

console.log(`Validated ${tuning.models.length} recipes and ${quickstart.profiles.length} quick-start profiles.`)
