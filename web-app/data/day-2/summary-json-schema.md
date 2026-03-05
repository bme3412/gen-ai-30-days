# JSON Schema Reference

**JSON Schema** is the standard for defining JSON structure. When working with LLM function calling and structured outputs, you use JSON Schema to specify exactly what parameters a function accepts or what structure an output should have.

The basic structure defines an object with **properties**. Each property has a **type**—primitives are `string`, `number`, `integer`, `boolean`, `null`, `array`, and `object`. List required properties in a separate **`required` array**, and set **`additionalProperties: false`** to prevent unlisted fields.

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0}
  },
  "required": ["name", "age"],
  "additionalProperties": false
}
```

**Strings** can be constrained with `minLength`, `maxLength`, regex `pattern`, and formats like `email`, `uri`, or `date-time`. **Numbers** support `minimum`, `maximum`, exclusive bounds, and `multipleOf`. These constraints guide the LLM toward valid values.

**Enums** restrict values to a fixed set—useful for status fields, priority levels, unit choices. The `const` keyword pins to exactly one value. **Arrays** define element type through `items` and can specify `minItems`, `maxItems`, and `uniqueItems`.

**Objects** can be nested to any depth, each with its own properties and constraints. For optional fields in strict mode, use a **null union**: `{"type": ["string", "null"]}`. The field is still required, but its value can be null.

**Composition keywords** combine schemas: `anyOf` accepts values matching any listed schema (union types), `oneOf` requires exactly one match, `allOf` requires all to match (merging constraints), `not` rejects matching values. **Conditional schemas** with `if/then/else` apply different rules based on field values.

**Reusable definitions** through `$defs` and `$ref` keep schemas DRY. Define common structures once under `$defs`, reference wherever needed—especially useful when the same structure (like an address) appears multiple times.

For LLM function calling with **strict mode**: require `additionalProperties: false` on all objects, all fields in `required`, and optional fields expressed as null unions.
