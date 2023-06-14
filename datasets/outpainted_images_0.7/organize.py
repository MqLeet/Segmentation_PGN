import os
import torch
import torchvision as tv
from tqdm import tqdm
from torchvision.transforms import Resize

black_img = torch.zeros(3, 512, 768, dtype=torch.uint8)


os.makedirs(name="list", exist_ok=True)
os.makedirs(name="labels", exist_ok=True)
os.makedirs(name="edges", exist_ok=True)

fv = open("list/val.txt", "w")
fi = open("list/val_id.txt", "w")

for d in tqdm(os.listdir("images")):
    n = d.split(".")[0]
    img = tv.io.read_image(f"images/{d}")
    img = img[:3, :, :]
    img = tv.transforms.Resize([512, 768])(img)
    print(img.shape)
    tv.io.write_png(img, f"images/{n}.png")
    tv.io.write_png(black_img, f"labels/{n}.png")
    tv.io.write_png(black_img, f"edges/{n}.png")

    fi.write(f"{n}\n")
    fv.write(f"/images/{n}.png /labels/{n}.png\n")

fi.close()
fv.close()
