import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Read Image ──────────────────────────────
image = cv2.imread(r"C:\Users\Student\Pictures\download.jpg")

# Check image loading
if image is None:
    print("Error: Image not found. Check the path.")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ── A. Brightness ───────────────────────────
bright = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * 96)

dark = cv2.subtract(image, np.ones(image.shape, dtype=np.uint8) * 96)


# ── B. Flip ─────────────────────────────────
h_flip = cv2.flip(image, 1)   # Horizontal flip
v_flip = cv2.flip(image, 0)   # Vertical flip


# ── C. Color Channels ───────────────────────
red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]


# ── D. Grayscale ────────────────────────────
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# ── E. Negative ─────────────────────────────
negative = 255 - image


# ── Display All Images ──────────────────────
plt.figure(figsize=(14, 10))


# Row 1 - Original and Brightness
plt.subplot(3, 4, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(bright)
plt.title("Brighter +96")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(dark)
plt.title("Darker -96")
plt.axis("off")


# Row 2 - Flip Images
plt.subplot(3, 4, 5)
plt.imshow(h_flip)
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(v_flip)
plt.title("Vertical Flip")
plt.axis("off")


# Row 2 - Color Channels
plt.subplot(6,8,7)
plt.imshow(red, cmap="Reds")
plt.title("Red Channel")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(green, cmap="Greens")
plt.title("Green Channel")
plt.axis("off")


# Row 3 - Remaining Images
plt.subplot(3, 4, 9)
plt.imshow(blue, cmap="Blues")
plt.title("Blue Channel")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(3, 4, 11)
plt.imshow(negative)
plt.title("Negative")
plt.axis("off")


plt.tight_layout()
plt.show()

