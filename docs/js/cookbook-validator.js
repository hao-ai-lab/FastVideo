/**
 * The cookbook's standard JSON Schema validator, shared by browser and Node tests.
 *
 * Ajv validates the authored JSON Schema 2020-12 field definitions. Compilation and
 * validation implement JSON Schema keywords, including nested objects, enums,
 * exclusive bounds, and if/then dependencies. No FastVideo rules live here.
 * Values come from the native baseline and explicit edits; validation never inserts defaults or changes data.
 *
 * The npm catalog-build and test commands build this source automatically.
 * `npm run build:validator --prefix docs` also builds it independently. The JS
 * bundle and license notices are Git-ignored outputs published together with
 * the site, with no CDN or Node requirement for readers.
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
