# 复现所用的四张真实照片

本目录图片取自 [scikit-image v0.25.2 样例数据](https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data)，由 [该版本的 SHA-256 注册表](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_registry.py)核验。下表的说明参考 [scikit-image 样例文档](https://scikit-image.org/docs/0.25.x/api/skimage.data.html)。

| 文件 | 内容 | 来源及授权 |
| --- | --- | --- |
| `astronaut.png` | Eileen Collins 宇航员肖像 | NASA，公有领域 |
| `chelsea.png` | Chelsea 猫 | Stefan van der Walt，CC0 |
| `coffee.png` | 咖啡杯 | Rachel Michetti，照片由 Pikolo Espresso Bar 提供，CC0 |
| `rocket.jpg` | Falcon 9 发射 | SpaceX 公共照片，公有领域 |

运行时通过 Pillow 读取图片，无需安装 scikit-image。文件哈希见 [实验清单](../../configs/reproduction_images.json)。
