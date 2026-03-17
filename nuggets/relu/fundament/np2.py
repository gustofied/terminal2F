# broadcasting
import numpy as np

broadcasters_inn = np.zeros((5, 8, 10))

print(broadcasters_inn)

broadcasters_inn[1, 4, 7] = 42

print(broadcasters_inn)

rng = np.random.default_rng(seed=42)
broadcasters_inn = rng.choice(
    400,
    size=(5, 8, 10),
    replace=False,
)

print(broadcasters_inn)

staff_cleaning = -1 * np.ones((5, 8, 10))
print(staff_cleaning)

print(broadcasters_inn * staff_cleaning )

print("--")

staff_cleaning = -1 * np.ones((1, 8, 10))
print(broadcasters_inn * staff_cleaning )
