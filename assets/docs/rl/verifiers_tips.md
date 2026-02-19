## Verifiers - Small Learnings

Things worth knowing when working with the verifiers library.

### Dataset auto-wrapping

If your dataset rows have a `question` string column, Verifiers will auto-wrap it into a user message for you:

```
# you provide:
"question": "What is 2+2?"

# it becomes:
prompt = [{"role": "user", "content": "What is 2+2?"}]
```

You don't need to manually build the messages list. Just give it a string and it handles the chat format.

### Environment IDs vs Hub references

Two different formats for two different commands:

- **`prime env install`** takes the full Hub reference: `owner/name@version`
  ```bash
  prime env install primeintellect/alphabet-sort@0.1.15
  ```
- **`vf-eval`** takes just the environment id: `alphabet-sort`
  ```bash
  prime eval run alphabet-sort -m gpt-4.1-mini -n 10
  ```

The `@0.1.15` is the Hub package version for installing. At eval time, it just needs the name - it imports the installed Python module (`alphabet_sort`) regardless of what version you installed.
