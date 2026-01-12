
# Process Lorem ipsum text
text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget finibus est, non pulvinar orci.
Phasellus dictum orci tellus, nec mollis arcu ornare ut. Etiam a risus convallis, mollis arcu nec, semper elit.
Nullam dolor est, commodo in purus a, mollis aliquam nibh. Nullam id massa at velit varius varius.
Maecenas vestibulum dictum vestibulum. Vestibulum sollicitudin et massa nec semper.
Phasellus mollis lacus in orci gravida tristique.
"""

# Split into sentences
sentences = text.strip().split("\n")

# Extract unique words (case-sensitive)
unique_words = set(word for sentence in sentences for word in sentence.split())

# Print results
print("=== Sentences ===")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

print("\n=== Unique Words ===")
print(unique_words)
