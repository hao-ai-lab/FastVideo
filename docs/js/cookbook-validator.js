/**
 * The cookbook's standard JSON Schema validator, shared by browser and Node tests.
 *
 * Ajv's 2020 class matches Pydantic's exported schema dialect. Compilation and
 * validation implement JSON Schema keywords, including nested objects, enums,
 * exclusive bounds, and if/then dependencies. No FastVideo rules live here.
 * Values come from the native baseline and explicit edits; validation never inserts defaults or changes data.
 *
 * Run `npm run build:validator --prefix docs` after editing this file or the
 * pinned dependencies. The result is a self-contained browser asset, with no
 * CDN or Node requirement for readers.
 */
import Ajv2020 from "ajv/dist/2020.js";

export function createValidator(schema) {
  // Ajv retains compiled schemas: each active catalog gets its own collectable instance.
  const ajv = new Ajv2020({
    allErrors: true,
    strict: true,
    ownProperties: true,
    coerceTypes: false,
    useDefaults: false,
    removeAdditional: false,
  });
  return ajv.compile(schema);
}
