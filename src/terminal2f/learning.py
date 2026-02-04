from dataclasses import dataclass

@dataclass(frozen=True)
class Comment:
    id: int
    text: str

hei = Comment(3, "hei")

print(hei)